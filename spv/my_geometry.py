from __future__ import division
import math
import random
import meep as mp
import numpy as np


class my_geometry:
    st_names = [
        '2D PhC',
        '2D PhC with substrate',
        '2D PhC with shift',
        'microdisk',
        'rough_microdisk',
        'waveguide_disk',
        'singular_bowtie_phc'
    ]
    # 同一パラメータでのジオメトリ再生成・重複printを防ぐクラスキャッシュ
    _geom_cache = {}

    def __init__(self, period, n):
        self.geometry = []
        self.prefix = ""
        self.period = period
        self.n = n
        self.st_dict = {i: name for i, name in enumerate(self.st_names)}

    def set_structure(self, geometry_index, *args):
        if geometry_index < len(self.st_dict) and (len(args) == 0 or args[0] != -1):
            cache_key = (self.period, self.n, geometry_index, repr(args))
            if cache_key in my_geometry._geom_cache:
                self.geometry, self.prefix = my_geometry._geom_cache[cache_key]
                return self.geometry, self.prefix
        else:
            cache_key = None

        if geometry_index >= len(self.st_dict):
            for i, name in self.st_dict.items():
                print(f"{i}:\t{name}")
                self.set_structure(i, -1)
                print()
        elif self.st_dict[geometry_index] == '2D PhC':
            self._2D_PhC(*args)
        elif self.st_dict[geometry_index] == '2D PhC with substrate':
            self._2D_PhC_with_substrate(*args)
        elif self.st_dict[geometry_index] == '2D PhC with shift':
            self._2D_PhC_with_shift(*args)
        elif self.st_dict[geometry_index] == 'microdisk':
            self.microdisk(*args)
        elif self.st_dict[geometry_index] == 'rough_microdisk':
            self.rough_microdisk(*args)
        elif self.st_dict[geometry_index] == 'waveguide_disk':
            self.waveguide_disk(*args)
        elif self.st_dict[geometry_index] == 'singular_bowtie_phc':
            self.singular_bowtie_phc(*args)
        else:
            self.geometry = []

        if cache_key is not None and len(self.geometry) > 0:
            my_geometry._geom_cache[cache_key] = (self.geometry, self.prefix)

        return self.geometry, self.prefix

    def _build_phc_holes(self, r, slab_t, Nx, Ny, defN, period, air):
        holes = []
        sqrt3_half_p = (math.sqrt(3) / 2.0) * period
        for j in range(-Ny, Ny + 1):
            y_pos = sqrt3_half_p * j
            is_odd_row = (abs(j) % 2 == 1)
            for i in range(-Nx, Nx):
                if j == 0 and abs(i) < defN:
                    continue
                x_pos = period * (i + 0.5) if is_odd_row else period * i
                holes.append(
                    mp.Cylinder(r, height=slab_t, center=mp.Vector3(x_pos, y_pos, 0), material=air)
                )
        return holes

    def _2D_PhC(self, *args):
        if len(args) == 0:
            self._2D_PhC(0.35, 0.4, 6, 5, 1)
            return
        elif len(args) == 5:
            r_ratio, slab_ratio, Nx, Ny, defN = args
            period, n = self.period, self.n
        else:
            print("1 r_ratio   :\t ratio of radius to the period")
            print("2 slab_ratio:\t ratio of slab thickness to the period")
            print("3 Nx        :\t number of the period in x axis")
            print("4 Ny        :\t number of the period in y axis")
            print("5 defN      :\t number of missing holes for defect cavity")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error _2D_PhC: geometry parameters are not set appropriately")
            return []

        r = r_ratio * period
        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)
        slab_t = round(slab_ratio * period, 3)

        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, slab_t), material=medium)
        ]
        geometry.extend(self._build_phc_holes(r, slab_t, Nx, Ny, defN, period, air))

        print("2D PhC")
        print(f"r_ratio   :\t {r_ratio}")
        print(f"slab_ratio:\t {slab_ratio}")
        print(f"Nx        :\t {Nx}")
        print(f"Ny        :\t {Ny}")
        print(f"defN      :\t {defN}")
        self.geometry = geometry
        self.prefix = f"2D({period:.2f} {n}, {r_ratio:.3f} {slab_ratio:.2f} {Nx} {Ny} {defN})"

    def _2D_PhC_with_substrate(self, *args):
        if len(args) == 0:
            self._2D_PhC_with_substrate(0.35, 0.4, 6, 5, 1, 0.25, 100)
            return
        elif len(args) == 7:
            r_ratio, slab_ratio, Nx, Ny, defN, sac_t, zm = args
            period, n = self.period, self.n
        else:
            print("1 r_ratio   :\t ratio of radius to the period")
            print("2 slab_ratio:\t ratio of slab thickness to the period")
            print("3 Nx        :\t number of the period in x axis")
            print("4 Ny        :\t number of the period in y axis")
            print("5 defN      :\t number of missing holes for defect cavity")
            print("6 sac_t     :\t ratio of the thickness of sacrificial layer to the period")
            print("7 zm        :\t length of analysis space in z axis")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error _2D_PhC_with_substrate: geometry parameters are not set appropriately")
            return []

        r = r_ratio * period
        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)
        slab_t = round(slab_ratio * period, 3)

        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, slab_t), material=medium)
        ]
        geometry.extend(self._build_phc_holes(r, slab_t, Nx, Ny, defN, period, air))
        geometry.append(
            mp.Block(center=mp.Vector3(0, 0, -(zm + slab_t) / 2 - sac_t), size=mp.Vector3(mp.inf, mp.inf, zm), material=medium)
        )

        print("2D PhC with substrate")
        print(f"r_ratio   :\t {r_ratio}")
        print(f"slab_ratio:\t {slab_ratio}")
        print(f"Nx        :\t {Nx}")
        print(f"Ny        :\t {Ny}")
        print(f"defN      :\t {defN}")
        print(f"sac_t     :\t {sac_t}")
        print(f"zm        :\t {zm}")
        self.geometry = geometry
        self.prefix = f"2DwSl({period:.2f} {n}, {r_ratio:.3f} {slab_ratio:.2f} {Nx} {Ny} {defN} {sac_t:.2f})"

    def _2D_PhC_with_shift(self, *args):
        if len(args) == 0:
            self._2D_PhC_with_shift(0.35, 0.4, 6, 5, 1, [0.3, 0.2, 0.1], [0.9, 1, 1])
            return
        elif len(args) == 7 and len(args[5]) == len(args[6]):
            r_ratio, slab_ratio, Nx, Ny, defN = args[:5]
            shift = np.array(args[5]) * self.period
            r_sh = np.array(args[6])
            period, n = self.period, self.n

            for i in range(len(shift)):
                if i != len(shift) - 1:
                    if 1 < (-args[5][i + 1] + args[5][i]) + r_ratio * (r_sh[i + 1] + r_sh[i]):
                        print("shift     :\t {}".format(str(shift).replace('[', '(').replace(']', ')')))
                        print("r_sh      :\t {}".format(str(r_sh).replace('[', '(').replace(']', ')')))
                        print("holes are overlapped")
                        return []
                else:
                    if 1 < args[5][i] + r_ratio * (1 + r_sh[i]):
                        print("shift     :\t {}".format(str(shift).replace('[', '(').replace(']', ')')))
                        print("r_sh      :\t {}".format(str(r_sh).replace('[', '(').replace(']', ')')))
                        print("holes are overlapped")
                        return []
        else:
            print("1 r_ratio   :\t ratio of radius to the period")
            print("2 slab_ratio:\t ratio of slab thickness to the period")
            print("3 Nx        :\t number of the period in x axis")
            print("4 Ny        :\t number of the period in y axis")
            print("5 defN      :\t number of missing holes for defect cavity")
            print("6 shift     :\t shift of the hole at the edges of the defect (divided by period)")
            print("7 r_sh      :\t change of the hole radius at the edges of the defect")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error _2D_PhC_with_shift: geometry parameters are not set appropriately")
            return []

        r = r_ratio * period
        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)
        slab_t = round(slab_ratio * period, 3)

        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, slab_t), material=medium)
        ]
        sqrt3_half_p = (math.sqrt(3) / 2.0) * period
        for j in range(-Ny, Ny + 1):
            for i in range(-Nx, Nx + 1):
                if j == 0 and abs(i) < defN:
                    continue
                if j == 0 and defN <= i < defN + len(shift):
                    idx = i - defN
                    geometry.append(
                        mp.Cylinder(r * r_sh[idx], height=slab_t, center=mp.Vector3(period * i + shift[idx], 0, 0), material=air)
                    )
                elif j == 0 and -(defN + len(shift)) < i <= -defN:
                    idx = -i - defN
                    geometry.append(
                        mp.Cylinder(r * r_sh[idx], height=slab_t, center=mp.Vector3(period * i - shift[idx], 0, 0), material=air)
                    )
                elif i != Nx:
                    x_pos = period * (i + 0.5) if abs(j) % 2 == 1 else period * i
                    geometry.append(
                        mp.Cylinder(r, height=slab_t, center=mp.Vector3(x_pos, sqrt3_half_p * j, 0), material=air)
                    )

        print("2D PhC with shift")
        print(f"r_ratio   :\t {r_ratio}")
        print(f"slab_ratio:\t {slab_ratio}")
        print(f"Nx        :\t {Nx}")
        print(f"Ny        :\t {Ny}")
        print(f"defN      :\t {defN}")
        print("shift     :\t {}".format(str(args[5]).replace('[', '(').replace(']', ')')))
        print("r_sh      :\t {}".format(str(r_sh).replace('[', '(').replace(']', ')')))
        self.geometry = geometry
        sh_str = ' '.join([f'{num:.3f}' for num in args[5]])
        rsh_str = ' '.join([f'{num:.3f}' for num in r_sh])
        self.prefix = f"2DwSh({period:.2f} {n}, {r_ratio:.3f} {slab_ratio:.2f} {Nx} {Ny} {defN} ({sh_str}) ({rsh_str}))"

    def microdisk(self, *args):
        if len(args) == 0:
            self.microdisk(1, 0.1, 0.3, 0.5, 3)
            return
        elif len(args) == 5:
            disk_D, disk_t, Sac_D, Sac_t, Out_D = args
            n = self.n
        else:
            print("1 disk_D   :\t disk diameter")
            print("2 disk_t   :\t disk thickness")
            print("3 Sac_D    :\t diameter of sacrificial layer")
            print("4 Sac_t    :\t thickness of sacrificial layer")
            print("5 Out_D    :\t diameter of outer circle ")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error microdisk: geometry parameters are not set appropriately")
            return []

        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)
        sac_z = -(disk_t / 2 + Sac_t / 2)

        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=air),
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, disk_t), material=medium),
            mp.Cylinder(Out_D / 2, height=mp.inf, center=mp.Vector3(0, 0, 0), material=air),
            mp.Block(center=mp.Vector3(0, 0, -(disk_t / 2 + Sac_t + 1.5)), size=mp.Vector3(mp.inf, mp.inf, 3), material=medium),
            mp.Cylinder(disk_D / 2, height=disk_t, center=mp.Vector3(0, 0, 0), material=medium),
            mp.Block(center=mp.Vector3(0, 0, sac_z), size=mp.Vector3(mp.inf, mp.inf, Sac_t), material=medium),
            mp.Cylinder((Out_D + disk_D - Sac_D) / 2, height=Sac_t, center=mp.Vector3(0, 0, sac_z), material=air),
            mp.Cylinder(Sac_D / 2, height=Sac_t, center=mp.Vector3(0, 0, sac_z), material=medium),
        ]

        print("microdisk")
        print(f"disk_D   :\t {disk_D}")
        print(f"disk_t   :\t {disk_t}")
        print(f"Sac_D        :\t {Sac_D}")
        print(f"Sac_t        :\t {Sac_t}")
        print(f"Out_D      :\t {Out_D}")
        self.geometry = geometry
        self.prefix = f"md({disk_D:.2f} {disk_t:.3f}, {n}, {Sac_D:.2f} {Sac_t:.3f}, {Out_D:.2f})"

    def rough_microdisk(self, *args):
        if len(args) == 0:
            self.rough_microdisk(1, 0.1, 0.3, 0.5, 3, 50 * 4, 0.01)
            return
        elif len(args) == 7:
            disk_D, disk_t, Sac_D, Sac_t, Out_D, Cir_reso, Dev_rate = args
            n = self.n
        else:
            print("1 disk_D   :\t disk diameter")
            print("2 disk_t   :\t disk thickness")
            print("3 Sac_D    :\t diameter of sacrificial layer")
            print("4 Sac_t    :\t thickness of sacrificial layer")
            print("5 Out_D    :\t diameter of outer circle ")
            print("6 Cir_reso    :\t angle resolution to approximate the disk circle, type: int")
            print("7 Dev_rate    :\t factor (or magnitude) of the deviation (-1 to 1) due to assumed fab error")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error rough_microdisk: geometry parameters are not set appropriately")
            return []

        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)

        N = Cir_reso
        delta = Dev_rate
        Nsub = 20
        M = 1000
        s = 2
        half_s = round(s / 2)

        print('dev list (fine) angle dependence of delta r')
        rad_seed = 0
        print('random seed: ' + str(rad_seed))
        random.seed(rad_seed)

        total_pts = N * Nsub
        dev = np.array([delta * random.uniform(-1, 1) for _ in range(total_pts)], dtype=np.float64)

        # 移動平均処理を高速化 (half_s == 1 のインプレース更新と完全等価な演算を保持しつつ高速化)
        if half_s == 1:
            for _ in range(M):
                for i in range(total_pts):
                    dev[i] = 0.5 * (dev[i - 1] + dev[i])
        else:
            for _ in range(M):
                for i in range(total_pts):
                    d = 0.0
                    for j in range(i - half_s, i + half_s):
                        d += dev[j % total_pts]
                    dev[i] = d / (2 * half_s)

        print('disk shape coordinates ')
        angles = np.radians(np.arange(N) * (360.0 / N))
        r_vals = (disk_D / 2.0) * (1.0 + dev[(np.arange(N) * Nsub) % total_pts])
        x_vals = r_vals * np.cos(angles)
        y_vals = r_vals * np.sin(angles)

        circle_list = [mp.Vector3(float(x), float(y), -disk_t / 2.0) for x, y in zip(x_vals, y_vals)]

        sac_z = -(disk_t / 2 + Sac_t / 2)
        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=air),
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, disk_t), material=medium),
            mp.Cylinder(Out_D / 2, height=mp.inf, center=mp.Vector3(0, 0, 0), material=air),
            mp.Block(center=mp.Vector3(0, 0, -(disk_t / 2 + Sac_t + 1.5)), size=mp.Vector3(mp.inf, mp.inf, 3), material=medium),
            mp.Prism(circle_list, height=disk_t, material=medium),
            mp.Block(center=mp.Vector3(0, 0, sac_z), size=mp.Vector3(mp.inf, mp.inf, Sac_t), material=medium),
            mp.Cylinder((Out_D + disk_D - Sac_D) / 2, height=Sac_t, center=mp.Vector3(0, 0, sac_z), material=air),
            mp.Cylinder(Sac_D / 2, height=Sac_t, center=mp.Vector3(0, 0, sac_z), material=medium),
        ]

        print("rough_microdisk")
        print(f"disk_D   :\t {disk_D}")
        print(f"disk_t   :\t {disk_t}")
        print(f"Sac_D        :\t {Sac_D}")
        print(f"Sac_t        :\t {Sac_t}")
        print(f"Out_D      :\t {Out_D}")
        print(f"Cir_reso      :\t {Cir_reso}")
        print(f"Dev_rate      :\t {Dev_rate}")
        self.geometry = geometry
        self.prefix = f"rmd({disk_D:.2f} {disk_t:.3f}, {n}, {Sac_D:.2f} {Sac_t:.3f}, {Out_D:.2f}, {Cir_reso} {Dev_rate:.3f})"

    def waveguide_disk(self, *args):
        if len(args) == 0:
            self.waveguide_disk(1, 0.1, 0.155, 0.1, 0.12, 3)
            return
        elif len(args) == 6:
            disk_D, disk_t, wg_w, wg_t, gap, Out_D = args
            n = self.n
        else:
            print("1 disk_D     :\t disk diameter")
            print("2 disk_t     :\t disk thickness")
            print("3 wg_w       :\t width of waveguide")
            print("4 wg_t       :\t thickness of waveguide")
            print("5 gap        :\t gap size between disk and waveguide")
            print("6 Out_D      :\t diameter of outer circle")
            if args[0] != -1:
                for i, arg in enumerate(args):
                    print(f"args[{i}]:\t{arg}")
                print("error waveguide_disk: geometry parameters are not set appropriately")
            return []

        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)

        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=air),
            mp.Cylinder(Out_D / 2, height=mp.inf, center=mp.Vector3(0, 0, 0), material=air),
            mp.Cylinder(disk_D / 2, height=disk_t, center=mp.Vector3(0, 0, 0), material=medium),
            mp.Block(
                center=mp.Vector3((disk_D / 2) + gap + (wg_w / 2), 0, 0),
                size=mp.Vector3(wg_w, mp.inf, wg_t),
                material=medium
            ),
        ]

        print("waveguide_disk")
        print(f"disk_D   :\t {disk_D}")
        print(f"disk_t   :\t {disk_t}")
        print(f"wg_w     :\t {wg_w}")
        print(f"wg_t     :\t {wg_t}")
        print(f"gap      :\t {gap}")
        print(f"Out_D    :\t {Out_D}")
        self.geometry = geometry
        self.prefix = f"wg_d({disk_D:.2f} {disk_t:.3f}, {wg_w:.2f} {wg_t:.2f}, {gap:.2f}, {n}, {Out_D:.2f})"

    def singular_bowtie_phc(self, *args):
        if len(args) == 0:
            # デフォルト値: r_ratio, slab_ratio, N_ring, twist_deg, core_r_ratio, gap_ratio, bowtie_angle
            self.singular_bowtie_phc(0.26, 0.5, 8, 3.0, 0.55, 0.01, 60.0)
            return
        elif len(args) == 7:
            r_ratio, slab_ratio, N_ring, twist_deg, core_r_ratio, gap_ratio, bowtie_angle = args
            period, n = self.period, self.n
        else:
            print("1 r_ratio      :\t 空気孔の半径 / period")
            print("2 slab_ratio   :\t スラブ厚 / period")
            print("3 N_ring       :\t 格子の周期数（中心からのリング数）")
            print("4 twist_deg    :\t ツイスト角 [度] (例: 2.0 ~ 6.0)")
            print("5 core_r_ratio :\t 中心ボウタイ領域の外径半径 / period")
            print("6 gap_ratio    :\t ボウタイ中心ギャップ幅 g / period")
            print("7 bowtie_angle :\t ボウタイ頂角 [度] (例: 60.0)")
            if args[0] != -1:
                print("error singular_bowtie_phc: geometry parameters are not set appropriately")
            return []

        r = r_ratio * period
        slab_t = round(slab_ratio * period, 3)
        core_r = core_r_ratio * period
        gap = gap_ratio * period
        half_gap = gap / 2.0
        half_theta = math.radians(twist_deg / 2.0)
        half_apex = math.radians(bowtie_angle / 2.0)

        medium = mp.Medium(epsilon=n**2)
        air = mp.Medium(epsilon=1)

        # 1. ベースとなる誘電体スラブ
        geometry = [
            mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, mp.inf, slab_t), material=medium)
        ]

        # 2. 2次元ツイスト格子（相対的に ±twist_deg/2 回転した2組の副格子）の生成
        a1 = np.array([period, 0.0])
        a2 = np.array([0.5 * period, (math.sqrt(3.0) / 2.0) * period])
        # ハニカム（グラフェン）副格子オフセット
        sub_offset = np.array([0.0, period / math.sqrt(3.0)])

        def rot_vec(vec, rad):
            c, s = math.cos(rad), math.sin(rad)
            return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c])

        for i in range(-N_ring, N_ring + 1):
            for j in range(-N_ring, N_ring + 1):
                base_pos = i * a1 + j * a2
                # 副格子A (-theta/2 回転) と 副格子B (+theta/2 回転)
                pos_A = rot_vec(base_pos - 0.5 * sub_offset, -half_theta)
                pos_B = rot_vec(base_pos + 0.5 * sub_offset, +half_theta)

                for p_xy in (pos_A, pos_B):
                    dist = math.hypot(p_xy[0], p_xy[1])
                    # 中心のボウタイ領域と干渉しない外側のみに空気孔を配置
                    if core_r + r < dist <= N_ring * period:
                        geometry.append(
                            mp.Cylinder(r, height=slab_t, center=mp.Vector3(float(p_xy[0]), float(p_xy[1]), 0), material=air)
                        )

        # 3. 中心欠陥部に円形の空気領域を配置
        geometry.append(
            mp.Cylinder(core_r, height=slab_t, center=mp.Vector3(0, 0, 0), material=air)
        )

        # 4. 誘電体ボウタイ・ナノアンテナ（x軸方向に対向する2つの三角形Prism）を配置
        # 外周（core_r * 1.1）まで伸ばして周囲のスラブと確実に接続させる
        x_outer = core_r * 1.1
        y_outer = (x_outer - half_gap) * math.tan(half_apex)

        right_bowtie_vertices = [
            mp.Vector3(half_gap, 0.0, -slab_t / 2.0),
            mp.Vector3(x_outer, y_outer, -slab_t / 2.0),
            mp.Vector3(x_outer, -y_outer, -slab_t / 2.0)
        ]
        left_bowtie_vertices = [
            mp.Vector3(-half_gap, 0.0, -slab_t / 2.0),
            mp.Vector3(-x_outer, -y_outer, -slab_t / 2.0),
            mp.Vector3(-x_outer, y_outer, -slab_t / 2.0)
        ]
        geometry.append(mp.Prism(right_bowtie_vertices, height=slab_t, material=medium))
        geometry.append(mp.Prism(left_bowtie_vertices, height=slab_t, material=medium))

        print("singular_bowtie_phc")
        print(f"r_ratio      :\t {r_ratio}")
        print(f"slab_ratio   :\t {slab_ratio}")
        print(f"N_ring       :\t {N_ring}")
        print(f"twist_deg    :\t {twist_deg}")
        print(f"core_r_ratio :\t {core_r_ratio}")
        print(f"gap_ratio    :\t {gap_ratio}")
        print(f"bowtie_angle :\t {bowtie_angle}")

        self.geometry = geometry
        self.prefix = (
            f"SBPhC({period:.3f} {n}, {r_ratio:.3f} {slab_ratio:.2f} "
            f"{N_ring} {twist_deg:.2f} {core_r_ratio:.2f} {gap_ratio:.4f} {bowtie_angle:.1f})"
        )

