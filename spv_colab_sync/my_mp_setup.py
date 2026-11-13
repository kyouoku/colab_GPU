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

try:
    from . import my_geometry as mg
except ImportError:
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
        self.px = px
        self.py = py
        self.pz = pz

        self.eps_averaging = False
        self.save_h5 = False
        self.cutoff_harminv = 5.0
        self.cutoff_mode = 5.0
        self.decay_dt = 50
        self.decay_by = 1e-5
        self.use_decay_stop = True
        self.src_seed = 42
        self.shot_id = None
        self.progress_tracker = None
        self.last_elapsed_sim_sec = 0.0
        self.last_stopped_meep_time = 0.0
        self.last_saved_pngs = []

        self._dft_slices = None
        self._dft_freqs = None
        self._last_sim = None

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
        last_print_time = [time.time()]

        def _step_func(sim):
            now = time.time()
            if now - last_print_time[0] >= interval_wall_sec:
                last_print_time[0] = now
                if self.progress_tracker is not None:
                    self.progress_tracker.report_in_step(sim.meep_time(), total_meep_time)

        return mp.at_every(5.0, _step_func)

    @staticmethod
    def _normalize_positions(src_pos):
        if isinstance(src_pos, np.ndarray):
            if src_pos.ndim == 1:
                return [src_pos.tolist()]
            return src_pos.tolist()
        if isinstance(src_pos, (list, tuple)) and len(src_pos) > 0:
            if isinstance(src_pos[0], (list, tuple, np.ndarray)):
                return [list(p) for p in src_pos]
            return [list(src_pos)]
        return [[0.0, 0.0, 0.0]]

    @staticmethod
    def _normalize_fields(sfield):
        if isinstance(sfield, str):
            return [sfield]
        return list(sfield)

    def _build_sources(self, src_pos, sfield, fcen, df, cutoff):
        pos_list = self._normalize_positions(src_pos)
        field_list = self._normalize_fields(sfield)
        n_total = len(pos_list) * len(field_list)
        rng = np.random.default_rng(self.src_seed)

        sources = []
        for pos in pos_list:
            pt = mp.Vector3(self.period * float(pos[0]), self.period * float(pos[1]), self.period * float(pos[2]))
            for sf in field_list:
                mp_f = self.field_name_to_mp_field(sf)
                if mp_f is None:
                    continue
                if n_total > 1:
                    phi = float(rng.uniform(0.0, 2.0 * math.pi))
                    amp = complex(math.cos(phi), math.sin(phi))
                else:
                    amp = 1.0
                sources.append(
                    mp.Source(
                        mp.GaussianSource(fcen, fwidth=df, cutoff=cutoff),
                        component=mp_f,
                        center=pt,
                        amplitude=amp
                    )
                )
        return sources

    def _select_decay_component(self, mfield_list, pt0):
        fields = self._normalize_fields(mfield_list)
        if abs(pt0.x) < 1e-9 and self.px in (1, -1):
            if self.pz == 1:
                preferred = 'ey' if self.px == 1 else 'ex'
            elif self.pz == -1:
                preferred = 'hx' if self.px == 1 else 'hy'
            else:
                preferred = fields[0]
            if preferred in fields:
                return self.field_name_to_mp_field(preferred)
        if abs(pt0.y) < 1e-9 and self.py in (1, -1):
            if self.pz == 1:
                preferred = 'ex' if self.py == 1 else 'ey'
            elif self.pz == -1:
                preferred = 'hy' if self.py == 1 else 'hx'
            else:
                preferred = fields[0]
            if preferred in fields:
                return self.field_name_to_mp_field(preferred)
        return self.field_name_to_mp_field(fields[0])

    def _build_decay_or_tmax_stop(self, mfield_list, pt0, tmax, total_meep_t):
        if not self.use_decay_stop:
            return tmax

        mp_c = self._select_decay_component(mfield_list, pt0)
        decay_cond = mp.stop_when_fields_decayed(
            self.decay_dt, mp_c, pt0, self.decay_by
        )

        def _stop_cond(sim):
            if decay_cond(sim):
                return True
            return sim.round_time() >= total_meep_t

        return _stop_cond

    def field_name_to_mp_field(self, field_name):
        if field_name in self._FIELD_MAP:
            return self._FIELD_MAP[field_name]
        print(f'component name is not matched: {field_name}')
        return None

    def field_name_to_mp_outputfield(self, field_name):
        if field_name in self._OUT_FIELD_MAP:
            return self._OUT_FIELD_MAP[field_name]
        print(f'component name is not matched: {field_name}')
        return None

    def get_spectral_fname(self, fcen, df, tmax):
        return f"S_fc{fcen}_df{df}_tm{tmax}{self.st_name}{self.init_name}"

    def get_harminv_fname(self, src_pos, m_pos, fcen, df, tmax, mfield_list=['ex'], sfield='hz'):
        pos_list = self._normalize_positions(src_pos)
        rep_pos = pos_list[0]
        sf_list = self._normalize_fields(sfield)
        rep_sf = sf_list[0]
        shot_tag = f"_s{self.shot_id}" if self.shot_id is not None else ""

        src_str = ' '.join([f'{float(num):.3f}' for num in rep_pos])
        self.file_name = f"H({fcen:.4f} {df:.4f} {tmax} ({src_str}){rep_sf}){self.st_name}{self.init_name}{shot_tag}"
        file_name_list = []
        for field in mfield_list:
            for pos in m_pos:
                mname = ' '.join([f'{num * self.period:.2f}' for num in pos])
                file_name_list.append(f"{self.file_name}_{field}({mname})")
        return file_name_list

    def get_mode_fname(self, src_pos, sfield, fcen, df, tmax, pos_xyz, out_field_list=['ex']):
        pos_list = self._normalize_positions(src_pos)
        rep_pos = pos_list[0]
        sf_list = self._normalize_fields(sfield)
        rep_sf = sf_list[0]

        src_str = ' '.join([f'{float(num):.3f}' for num in rep_pos])
        return f"M({fcen:.4f} {df:.4f} {tmax} ({src_str}){rep_sf}){self.st_name}{self.init_name}"

    def get_freq_solver_fname(self, src_pos, df, fcen, out_field_list=['ex']):
        pos_list = self._normalize_positions(src_pos)
        rep_pos = pos_list[0]
        src_str = ' '.join([f'{float(num):.3f}' for num in rep_pos])
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
        cache_key = (self.st_name, self.resolution, float(self.cell.x), float(self.cell.y), float(self.cell.z))
        if cache_key in my_mp_setup._checked_eps_cache:
            print(f"[高速化スキップ] check_epsilon は同一条件 ({self.st_name}, res={self.resolution}) で出力済みのため省略します。")
            return []
        my_mp_setup._checked_eps_cache.add(cache_key)

        saved_paths = []

        def save_slice_fig(slice_data, base_fname, plane, name):
            if not mp.am_master():
                return
            fig, ax = plt.subplots()
            ax.imshow(slice_data.T, interpolation='bilinear', cmap='binary')
            out_path = f"{base_fname}_{plane}_{name}.png"
            fig.savefig(out_path)
            plt.close(fig)
            saved_paths.append(out_path)

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

            eps_yz = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(0, self.cell.y, self.cell.z)), component=mp.Dielectric)
            eps_xz = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(self.cell.x, 0, self.cell.z)), component=mp.Dielectric)
            eps_xy = sim.get_array(vol=mp.Volume(mp.Vector3(0), size=mp.Vector3(self.cell.x, self.cell.y, 0)), component=mp.Dielectric)

            save_slice_fig(eps_yz, file_name, 'yz', 'eps')
            save_slice_fig(eps_xz, file_name, 'xz', 'eps')
            save_slice_fig(eps_xy, file_name, 'xy', 'eps')
        else:
            print("Error: structure is not set")
        return saved_paths

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

    def harminv_calc(self, src_pos, m_pos, fcen, df, tmax, mfield_list=['ex'], sfield='hz', capture_dft=False, dft_nfreq=11):
        def harminv_output(file_name_list, h_setup, field_list, mpos_list):
            if not mp.am_master():
                return
            for i in range(len(field_list)):
                for j in range(len(mpos_list)):
                    idx = i * len(mpos_list) + j
                    path_w = file_name_list[idx] + '.txt'
                    modes = h_setup[idx].modes
                    with open(path_w, mode='w') as f:
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
            src = self._build_sources(src_pos, sfield, fcen, df, self.cutoff_harminv)
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

            self._dft_slices = None
            self._dft_freqs = None
            self._last_sim = None
            if capture_dft:
                dft_comps = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
                slice_specs = [
                    ('z0_xy', mp.Volume(mp.Vector3(0, 0, 0), size=mp.Vector3(self.cell.x, self.cell.y, 0))),
                    ('-z_xy', mp.Volume(mp.Vector3(0, 0, -0.06), size=mp.Vector3(self.cell.x, self.cell.y, 0))),
                    ('xz',    mp.Volume(mp.Vector3(0, 0, 0), size=mp.Vector3(self.cell.x, 0, self.cell.z))),
                    ('yz',    mp.Volume(mp.Vector3(0, 0, 0), size=mp.Vector3(0, self.cell.y, self.cell.z))),
                ]
                self._dft_slices = []
                for suffix, vol in slice_specs:
                    dft_obj = sim.add_dft_fields(dft_comps, fcen, df, dft_nfreq, where=vol)
                    self._dft_slices.append((suffix, vol, dft_obj))
                if dft_nfreq > 1:
                    self._dft_freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, dft_nfreq)
                else:
                    self._dft_freqs = np.array([fcen])

            sim.init_sim()
            order = list(h_order)

            if self.save_h5:
                temp_name = self.file_name + "_({})".format(' '.join([f'{num:.2f}' for num in mpos_list[0]]))
                order.append(
                    mp.after_sources(
                        mp.at_every(1.0 / self.resolution, mp.to_appended(temp_name, mp.in_point(mpos_list[0], mp.output_hfield)))
                    )
                )

            total_meep_t = (2.0 * self.cutoff_harminv / df) + tmax
            order.append(self._create_progress_step_func(total_meep_t))

            stop_cond = self._build_decay_or_tmax_stop(mfield_list, mpos_list[0], tmax, total_meep_t)
            t0_wall = time.time()
            sim.run(*order, until_after_sources=stop_cond)
            self.last_elapsed_sim_sec = time.time() - t0_wall
            self.last_stopped_meep_time = float(sim.meep_time())

            if mp.am_master():
                print(
                    f"[Harminv終了] Meep時間={self.last_stopped_meep_time:.1f} / 上限{total_meep_t:.1f} "
                    f"(実時間={self.last_elapsed_sim_sec:.1f}s)"
                )

            harminv_output(file_name, h_setup, mfield_list, mpos_list)
            if capture_dft:
                self._last_sim = sim
        else:
            print("Error: structure is not set")
            return [f + '.txt' for f in file_name]

    @staticmethod
    def _save_overlay_fig(eps_data, f_data, name, cmap_name):
        if not mp.am_master():
            return None
        fig, ax = plt.subplots()
        ax.imshow(eps_data.T, interpolation='bilinear', cmap='binary')
        im = ax.imshow(np.real(f_data).T, interpolation='bilinear', cmap=cmap_name, alpha=0.9)
        fig.colorbar(im, ax=ax)
        out_png = name + ".png"
        fig.savefig(out_png)
        plt.close(fig)
        return out_png

    def save_dft_mode_profile(self, target_freq, src_pos, sfield, df_mode, tmax_mode, pos_xyz=None, out_field_list=['ex']):
        if self._last_sim is None or not self._dft_slices or self._dft_freqs is None:
            return []

        sim = self._last_sim
        freq_idx = int(np.argmin(np.abs(self._dft_freqs - float(target_freq))))
        file_name = self.get_mode_fname(src_pos, sfield, target_freq, df_mode, tmax_mode, pos_xyz, out_field_list)
        print(f"start DFT mode output (merged STEP 2.5+3, freq_bin={self._dft_freqs[freq_idx]:.5f}): {file_name}")

        mp.all_wait()
        saved_pngs = []
        for suffix, vol, dft_obj in self._dft_slices:
            eps_data = sim.get_array(vol=vol, component=mp.Dielectric)
            ex = sim.get_dft_array(dft_obj, mp.Ex, freq_idx)
            ey = sim.get_dft_array(dft_obj, mp.Ey, freq_idx)
            ez = sim.get_dft_array(dft_obj, mp.Ez, freq_idx)
            hx = sim.get_dft_array(dft_obj, mp.Hx, freq_idx)
            hy = sim.get_dft_array(dft_obj, mp.Hy, freq_idx)
            hz = sim.get_dft_array(dft_obj, mp.Hz, freq_idx)

            e_sq = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
            h_sq = np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2
            dden_data = 0.5 * eps_data * e_sq
            hden_data = 0.5 * h_sq

            comment = file_name + suffix
            p1 = self._save_overlay_fig(eps_data, dden_data, comment + '-Dden', 'Reds')
            p2 = self._save_overlay_fig(eps_data, hden_data, comment + '-Hden', 'Reds')
            if p1:
                saved_pngs.append(p1)
            if p2:
                saved_pngs.append(p2)

        self._last_sim = None
        self._dft_slices = None
        self.last_saved_pngs = saved_pngs
        return saved_pngs

    def mode_calc(self, src_pos, sfield, fcen, df, tmax, pos_xyz, out_field_list=['ex'], decay_pt=None):
        saved_pngs = []

        def _2D_field_out(sim, file_name):
            def out_field(pos, v, comment):
                sim_vol = mp.Volume(mp.Vector3(*pos), size=mp.Vector3(*v))
                eps_data = sim.get_array(vol=sim_vol, component=mp.Dielectric)
                Dden_data = sim.get_array(vol=sim_vol, component=mp.D_EnergyDensity)
                Hden_data = sim.get_array(vol=sim_vol, component=mp.H_EnergyDensity)
                p1 = self._save_overlay_fig(eps_data, Dden_data, comment + '-Dden', 'Reds')
                p2 = self._save_overlay_fig(eps_data, Hden_data, comment + '-Hden', 'Reds')
                if p1:
                    saved_pngs.append(p1)
                if p2:
                    saved_pngs.append(p2)

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
            src = self._build_sources(src_pos, sfield, fcen, df, self.cutoff_mode)
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

            if decay_pt is not None:
                pt0 = mp.Vector3(self.period * decay_pt[0], self.period * decay_pt[1], self.period * decay_pt[2])
            else:
                pos_list = self._normalize_positions(src_pos)
                pt0 = mp.Vector3(self.period * pos_list[0][0], self.period * pos_list[0][1], self.period * pos_list[0][2])

            stop_cond = self._build_decay_or_tmax_stop(self._normalize_fields(sfield), pt0, tmax, total_meep_t)
            t0_wall = time.time()
            sim.run(*run_args, until_after_sources=stop_cond)
            self.last_elapsed_sim_sec = time.time() - t0_wall
            self.last_stopped_meep_time = float(sim.meep_time())

            _2D_field_out(sim, file_name)
            self.last_saved_pngs = saved_pngs
            return saved_pngs
        else:
            print("Error: structure is not set")
            return []
