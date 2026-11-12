import glob
import math
import os
import re
import time
import matplotlib
matplotlib.use('Agg')  # MPI並列・バッチ実行時の高速化とクラッシュ防止
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import my_geometry as mg


class my_mp_setup:
    _FIELD_MAP = {
        'ex': mp.Ex, 'ey': mp.Ey, 'ez': mp.Ez,
        'hx': mp.Hx, 'hy': mp.Hy, 'hz': mp.Hz
    }
    _OUT_FIELD_MAP = {
        'ex': mp.output_efield_x, 'ey': mp.output_efield_y, 'ez': mp.output_efield_z,
        'hx': mp.output_hfield_x, 'hy': mp.output_hfield_y, 'hz': mp.output_hfield_z
    }
    # 同一構造・同一解像度での check_epsilon() 重複実行を防止するキャッシュ
    _checked_eps_cache = set()

    def __init__(self, p_name, prefix, resolution, xm, ym, zm, pml_w, px, py, pz, period, n):
        self.p_name = os.path.basename(p_name).split('.')[0]
        self.file_name = ''
        self.geometry = []
        self.structure_prefix = ''
        self.st_name = ''
        self.sim_name = ''
        self.sym = []

        self.prefix = prefix
        self.resolution = resolution
        self.pml_w = pml_w
        self.period = period
        self.n = n

        # 高速化・進捗表示用の属性
        self.eps_averaging = False
        self.save_h5 = False
        self.cutoff_harminv = 4.0
        self.cutoff_mode = 3.0
        self.progress_tracker = None

        self.cell = mp.Vector3(xm, ym, zm)
        self.my_geom = mg.my_geometry(period, n)

        if px in (1, -1):
            self.sym.append(mp.Mirror(mp.X, phase=px))
        if py in (1, -1):
            self.sym.append(mp.Mirror(mp.Y, phase=py))
        if pz in (1, -1):
            self.sym.append(mp.Mirror(mp.Z, phase=pz))

        self.pml_layers = [
            mp.PML(pml_w[0], direction=mp.X),
            mp.PML(pml_w[1], direction=mp.Y),
            mp.PML(pml_w[2], direction=mp.Z)
        ]

        p_dict = {1: '+', -1: '-', 0: 'o'}
        pml_str = ' '.join([f'{num:.2f}' for num in pml_w])
        self.init_name = (
            f"_({xm:.1f} {ym:.1f} {zm:.1f} [{pml_str}] {resolution} "
            f"{p_dict[px]}{p_dict[py]}{p_dict[pz]})_{self.prefix}"
        )
        print(self.init_name)

    def _create_progress_step_func(self, total_meep_time, interval_wall_sec=30.0):
        """シミュレーション中に約30秒おきに残り時間と終了予定時刻を出力するステップ関数"""
        last_print_time = [time.time()]

        def _step_func(sim):
            now = time.time()
            if now - last_print_time[0] >= interval_wall_sec:
                last_print_time[0] = now
                if self.progress_tracker is not None:
                    self.progress_tracker.report_in_step(sim.meep_time(), total_meep_time)

        # ★ 分類A-⑤: C++からPythonへのコールバック頻度を 1.0 ➔ 5.0 Meep時間ごとに緩和してオーバーヘッドを削減
        return mp.at_every(5.0, _step_func)

    def field_name_to_mp_field(self, field_name):
        if field_name in self._FIELD_MAP:
            return self._FIELD_MAP[field_name]
        print('component name is not matched')
        return None

    def field_name_to_mp_outputfield(self, field_name):
        if field_name in self._OUT_FIELD_MAP:
            return self._OUT_FIELD_MAP[field_name]
        print('component name is not matched')
        return None

    def get_spectral_fname(self, fcen, df, tmax):
        return f"S_fc{fcen}_df{df}_tm{tmax}{self.st_name}{self.init_name}"

    def get_harminv_fname(self, src_pos, m_pos, fcen, df, tmax, mfield_list=['ex'], sfield='hz'):
        src_str = ' '.join([f'{num:.3f}' for num in src_pos])
        self.file_name = f"H({fcen:.4f} {df:.4f} {tmax} ({src_str}){sfield}){self.st_name}{self.init_name}"
        file_name_list = []
        for field in mfield_list:
            for pos in m_pos:
                mname = ' '.join([f'{num * self.period:.2f}' for num in pos])
                file_name_list.append(f"{self.file_name}_{field}({mname})")
        return file_name_list

    def get_mode_fname(self, src_pos, sfield, fcen, df, tmax, pos_xyz, out_field_list=['ex']):
        src_str = ' '.join([f'{num:.3f}' for num in src_pos])
        return f"M({fcen:.4f} {df:.4f} {tmax} ({src_str}){sfield}){self.st_name}{self.init_name}"

    def get_freq_solver_fname(self, src_pos, df, fcen, out_field_list=['ex']):
        src_str = ' '.join([f'{num:.3f}' for num in src_pos])
        return f"E({fcen:.4f} {df:.4f} [{src_str}]){self.st_name}{self.init_name}"

    def reset_geometry(self):
        self.geometry = []

    def set_structure(self, geometry_index, *args):
        self.geometry, self.st_name = self.my_geom.set_structure(geometry_index, *args)

    def remove_glob(self, pathname, recursive=True):
        if mp.am_master():
            for p in glob.glob(pathname, recursive=recursive):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def check_epsilon(self):
        # 同じ構造・解像度・セルサイズであれば2回目以降の check_epsilon をスキップして高速化
        cache_key = (self.st_name, self.resolution, float(self.cell.x), float(self.cell.y), float(self.cell.z))
        if cache_key in my_mp_setup._checked_eps_cache:
            print(f"[高速化スキップ] check_epsilon は同一条件 ({self.st_name}, res={self.resolution}) で出力済みのため省略します。")
            return
        my_mp_setup._checked_eps_cache.add(cache_key)

        def save_slice_fig(slice_data, base_fname, plane, name):
            if not mp.am_master():
                return
            fig, ax = plt.subplots()
            ax.imshow(slice_data.T, interpolation='bilinear', cmap='binary')
            prefix_name = self.file_name if self.file_name else base_fname
            fig.savefig(f"{prefix_name}_{plane}_{name}.png")
            plt.close(fig)

        if len(self.geometry) != 0:
            file_name = "epchk" + self.st_name + self.init_name
            print("start:" + file_name)

            sim = mp.Simulation(
                cell_size=self.cell,
                geometry=self.geometry,
                boundary_layers=self.pml_layers,
                symmetries=self.sym,
                resolution=self.resolution,
                eps_averaging=False
            )
            sim.init_sim()

            # ★ 分類A-⑤: 3次元セル全体ではなく、必要な2次元断面だけを直接取得してメモリと通信時間を大幅削減
            eps_yz = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(0, self.cell.y, self.cell.z)), component=mp.Dielectric)
            eps_xz = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(self.cell.x, 0, self.cell.z)), component=mp.Dielectric)
            eps_xy = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(self.cell.x, self.cell.y, 0)), component=mp.Dielectric)

            save_slice_fig(eps_yz, file_name, 'yz', 'eps')
            save_slice_fig(eps_xz, file_name, 'xz', 'eps')
            save_slice_fig(eps_xy, file_name, 'xy', 'eps')
        else:
            print("Error: structure is not set")

    def spectral_calc(self, fcen, df, tmax):
        file_name = self.get_spectral_fname(fcen, df, tmax)
        print("start:" + file_name)
        pt = mp.Vector3(-self.period * 1.3, -self.period * 1.0, -self.period * 0.4 * 0.3)
        src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Hz, pt)]

        if len(self.geometry) != 0:
            sim = mp.Simulation(
                cell_size=self.cell,
                geometry=self.geometry,
                boundary_layers=self.pml_layers,
                sources=src,
                symmetries=self.sym,
                resolution=self.resolution,
                eps_averaging=self.eps_averaging,
                subpixel_tol=1e-2,
                subpixel_maxeval=1000
            )
            sim.run(
                mp.at_beginning(mp.output_epsilon),
                mp.at_every(1.0 / self.resolution, mp.to_appended(file_name, mp.in_point(pt, mp.output_hfield_z))),
                until_after_sources=tmax
            )
        else:
            print("Error: structure is not set")
            return file_name

    def harminv_calc(self, src_pos, m_pos, fcen, df, tmax, mfield_list=['ex'], sfield='hz'):
        def harminv_output(file_name_list, h_setup, field_list, mpos_list):
            if not mp.am_master():
                return
            for i in range(len(field_list)):
                for j in range(len(mpos_list)):
                    idx = i * len(mpos_list) + j
                    path_w = file_name_list[idx] + '.txt'
                    modes = h_setup[idx].modes
                    with open(path_w, mode='a') as f:
                        for m_idx, m_obj in enumerate(modes):
                            mode_list = re.findall(r'Mode\((.*)\)', str(m_obj))
                            if not mode_list:
                                continue
                            mode = mode_list[0].split(',')
                            if m_idx == 0:
                                f.write('\t'.join(p.split('=')[0] for p in mode) + '\t\n')
                            f.write('\t'.join(p.split('=')[1].translate(str.maketrans({'(': '', ')': ''})) for p in mode) + '\t\n')
                        f.write('\n')

        file_name = self.get_harminv_fname(src_pos, m_pos, fcen, df, tmax, mfield_list, sfield)
        print("start harminv")
        for fname in file_name:
            print(fname)

        if len(self.geometry) != 0:
            src = [
                mp.Source(
                    mp.GaussianSource(fcen, fwidth=df, cutoff=self.cutoff_harminv),
                    self.field_name_to_mp_field(sfield),
                    mp.Vector3(self.period * src_pos[0], self.period * src_pos[1], self.period * src_pos[2]),
                    amplitude=1
                )
            ]
            # ★ 分類A-②: subpixel_tol=1e-2, subpixel_maxeval=1000 により高解像度でのサブピクセル初期化を高速化
            sim = mp.Simulation(
                cell_size=self.cell,
                geometry=self.geometry,
                boundary_layers=self.pml_layers,
                sources=src,
                symmetries=self.sym,
                resolution=self.resolution,
                eps_averaging=self.eps_averaging,
                subpixel_tol=1e-2,
                subpixel_maxeval=1000
            )

            mpos_list = [mp.Vector3(self.period * p[0], self.period * p[1], self.period * p[2]) for p in m_pos]
            h_setup = []
            h_order = []
            for f_name in mfield_list:
                mp_f = self.field_name_to_mp_field(f_name)
                for pos_vec in mpos_list:
                    h_inst = mp.Harminv(mp_f, pos_vec, fcen, df)
                    h_setup.append(h_inst)
                    h_order.append(mp.after_sources(h_inst))

            sim.init_sim()
            order = list(h_order)

            # save_h5 が True の場合のみ毎ステップの磁場HDF5書き出しを行う (デフォルトFalseでI/O高速化)
            if self.save_h5:
                temp_name = self.file_name + "_({})".format(' '.join([f'{num:.2f}' for num in mpos_list[0]]))
                order.append(
                    mp.after_sources(
                        mp.at_every(1.0 / self.resolution, mp.to_appended(temp_name, mp.in_point(mpos_list[0], mp.output_hfield)))
                    )
                )

            total_meep_t = (2.0 * self.cutoff_harminv / df) + tmax
            order.append(self._create_progress_step_func(total_meep_t))

            sim.run(*order, until_after_sources=tmax)
            harminv_output(file_name, h_setup, mfield_list, mpos_list)
        else:
            print("Error: structure is not set")
            return [f + '.txt' for f in file_name]

    def mode_calc(self, src_pos, sfield, fcen, df, tmax, pos_xyz, out_field_list=['ex']):
        def show_fig(eps_data, f_data, name, cmap_name):
            if not mp.am_master():
                return
            fig, ax = plt.subplots()
            ax.imshow(eps_data.T, interpolation='bilinear', cmap='binary')
            im = ax.imshow(np.real(f_data).T, interpolation='bilinear', cmap=cmap_name, alpha=0.9)
            fig.colorbar(im, ax=ax)
            fig.savefig(name + ".png")
            plt.close(fig)

        def _2D_field_out(sim, file_name):
            def out_field(pos, v, comment):
                sim_vol = mp.Volume(mp.Vector3(*pos), size=mp.Vector3(*v))
                eps_data = sim.get_array(vol=sim_vol, component=mp.Dielectric)
                Dden_data = sim.get_array(vol=sim_vol, component=mp.D_EnergyDensity)
                Hden_data = sim.get_array(vol=sim_vol, component=mp.H_EnergyDensity)
                show_fig(eps_data, Dden_data, comment + '-Dden', 'Reds')
                show_fig(eps_data, Hden_data, comment + '-Hden', 'Reds')

            # ★ 分類A-⑤: save_h5=False のときは不要な追加ステップ実行 (2.0 / resolution) を省略
            if self.save_h5:
                sim.run(
                    mp.at_end(mp.to_appended(file_name + '_h', mp.output_hfield)),
                    mp.at_end(mp.to_appended(file_name + '_e', mp.output_efield)),
                    until=2.0 / self.resolution
                )
            mp.all_wait()

            pos = [0, 0, 0]
            out_field(pos, [self.cell[0], self.cell[1], 0], file_name + 'z0_xy')
            out_field([0, 0, -0.06], [self.cell[0], self.cell[1], 0], file_name + '-z_xy')
            out_field(pos, [self.cell[0], 0, self.cell[2]], file_name + 'xz')
            out_field(pos, [0, self.cell[1], self.cell[2]], file_name + 'yz')

        file_name = self.get_mode_fname(src_pos, sfield, fcen, df, tmax, pos_xyz, out_field_list)
        print("start:" + file_name)

        if len(self.geometry) != 0:
            src = [
                mp.Source(
                    mp.GaussianSource(fcen, fwidth=df, cutoff=self.cutoff_mode),
                    self.field_name_to_mp_field(sfield),
                    mp.Vector3(self.period * src_pos[0], self.period * src_pos[1], self.period * src_pos[2])
                )
            ]
            # ★ 分類A-②: subpixel_tol=1e-2, subpixel_maxeval=1000 を指定
            sim = mp.Simulation(
                cell_size=self.cell,
                geometry=self.geometry,
                boundary_layers=self.pml_layers,
                sources=src,
                symmetries=self.sym,
                resolution=self.resolution,
                eps_averaging=self.eps_averaging,
                subpixel_tol=1e-2,
                subpixel_maxeval=1000
            )

            total_meep_t = (2.0 * self.cutoff_mode / df) + tmax
            run_args = [self._create_progress_step_func(total_meep_t)]
            if self.save_h5:
                v = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(self.cell[0], self.cell[1], self.cell[2]))
                run_args.insert(0, mp.at_beginning(mp.in_volume(v, mp.output_epsilon)))

            sim.run(*run_args, until_after_sources=tmax)
            _2D_field_out(sim, file_name)
        else:
            print("Error: structure is not set")
            return file_name

    def freq_solver(self, src_pos, df, fcen, out_field_list=['ex']):
        def show_fig(eps_data, f_data, base_fname, name):
            if not mp.am_master():
                return
            fig, ax = plt.subplots()
            ax.imshow(eps_data.T, interpolation='bilinear', cmap='binary')
            ax.imshow(np.real(f_data).T, interpolation='bilinear', cmap='RdBu', alpha=0.9)
            ax.axis('off')
            fig.savefig(f"{base_fname}_{name}.png")
            plt.close(fig)

        file_name = self.get_freq_solver_fname(src_pos, df, fcen, out_field_list)
        if len(self.geometry) != 0:
            print("start:" + file_name)
            src_sl = 10
            src = [
                mp.Source(
                    mp.ContinuousSource(frequency=fcen, fwidth=df, cutoff=src_sl),
                    component=mp.Ey,
                    center=mp.Vector3(self.period * src_pos[0], self.period * src_pos[1], self.period * src_pos[2])
                )
            ]
            sim = mp.Simulation(
                cell_size=self.cell,
                geometry=self.geometry,
                boundary_layers=self.pml_layers,
                sources=src,
                force_complex_fields=True,
                symmetries=self.sym,
                resolution=self.resolution,
                eps_averaging=self.eps_averaging,
                subpixel_tol=1e-2,
                subpixel_maxeval=1000
            )

            tol = 1e-7
            max_itr = 100
            L = 10
            guess_freq = fcen
            cw_tol = 1e-10
            cw_max_itr = 10000

            sim.init_sim()

            if mp.am_master():
                with open(f'./{file_name}_set.txt', mode='w') as f:
                    f.write("")

            eigfreq = sim.solve_eigfreq(
                tol=tol, maxiters=max_itr, guessfreq=guess_freq, cwtol=cw_tol, cwmaxiters=cw_max_itr, L=L
            )

            if mp.am_master():
                q_val = eigfreq.real / (-2 * eigfreq.imag)
                with open(f'./{file_name}_Q.txt', mode='a') as f:
                    f.write(f"eigfreq:\t{file_name}{fcen}\t{eigfreq.real}\t{q_val}\n")

            sim.run(until=2.0 / self.resolution)

            sim_vol = mp.Volume(mp.Vector3(0), size=mp.Vector3(self.cell[0], self.cell[1], 0))
            eps_data = sim.get_array(vol=sim_vol, component=mp.Dielectric)
            ex_data = sim.get_array(vol=sim_vol, component=mp.Ex)
            ey_data = sim.get_array(vol=sim_vol, component=mp.Ey)
            hz_data = sim.get_array(vol=sim_vol, component=mp.Hz)

            show_fig(eps_data, ex_data, file_name, 'ex')
            show_fig(eps_data, ey_data, file_name, 'ey')
            show_fig(eps_data, hz_data, file_name, 'hz')
        else:
            print("Error: structure is not set")
            return file_name

