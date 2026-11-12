"""
Photonic cavity structures ported from spv/my_geometry.py to JAX SDF -> eps(x, y, z) (Spec B1 §7).
Supports structure_type:
  0 : 2D PhC defect cavity (for V2 validation)
  3 : microdisk
  4 : rough_microdisk
  5 : waveguide_disk
  6 : singular_bowtie_phc
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Tuple
import jax.numpy as jnp
import numpy as np

from ..config import normalize_cfg
from .shapes import (
    sdf_box,
    sdf_cylinder_z,
    sdf_extruded_star_polygon_z,
    sdf_extruded_triangle_z,
    sdf_multi_cylinders_xy,
    sdf_slab_z,
)
from .smoothing import evaluate_permittivity_grid


def _generate_twisted_bowtie_phc_holes(
    period: float,
    r: float,
    n_ring: int,
    twist_deg: float,
    core_r: float,
) -> np.ndarray:
    """
    spv/my_geometry.py の singular_bowtie_phc (L447-470) と完全同一の
    2次元ツイスト・ハニカム格子（±twist_deg/2 回転した副格子A・B）の空気孔座標リストを生成する。
    """
    half_theta = math.radians(twist_deg / 2.0)
    a1 = np.array([period, 0.0], dtype=np.float64)
    a2 = np.array([0.5 * period, (math.sqrt(3.0) / 2.0) * period], dtype=np.float64)
    sub_offset = np.array([0.0, period / math.sqrt(3.0)], dtype=np.float64)

    def rot_vec(vec: np.ndarray, rad: float) -> np.ndarray:
        c, s = math.cos(rad), math.sin(rad)
        return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c], dtype=np.float64)

    holes: List[Tuple[float, float]] = []
    r_min = core_r + r
    r_max = n_ring * period

    for i in range(-n_ring, n_ring + 1):
        for j in range(-n_ring, n_ring + 1):
            base_pos = i * a1 + j * a2
            # 副格子A (-theta/2 回転) と 副格子B (+theta/2 回転)
            pos_a = rot_vec(base_pos - 0.5 * sub_offset, -half_theta)
            pos_b = rot_vec(base_pos + 0.5 * sub_offset, +half_theta)

            for p_xy in (pos_a, pos_b):
                dist = math.hypot(float(p_xy[0]), float(p_xy[1]))
                # 中心のボウタイ領域と干渉しない外側のみに空気孔を配置
                if r_min < dist <= r_max:
                    holes.append((float(p_xy[0]), float(p_xy[1])))

    if not holes:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(holes, dtype=np.float32)


def _generate_triangular_phc_holes(
    period: float,
    nx: int,
    ny: int,
    def_n: int,
) -> np.ndarray:
    """
    spv/my_geometry.py の _build_phc_holes (L64-76) と完全同一の三角格子空気孔座標リストを生成する。
    """
    holes: List[Tuple[float, float]] = []
    sqrt3_half_p = (math.sqrt(3.0) / 2.0) * period
    for j in range(-ny, ny + 1):
        y_pos = sqrt3_half_p * j
        is_odd_row = (abs(j) % 2 == 1)
        for i in range(-nx, nx):
            if j == 0 and abs(i) < def_n:
                continue
            x_pos = period * (i + 0.5) if is_odd_row else period * i
            holes.append((float(x_pos), float(y_pos)))
    if not holes:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(holes, dtype=np.float32)


def _generate_rough_disk_radii(disk_d: float, cir_reso: int, dev_rate: float) -> np.ndarray:
    """
    spv/my_geometry.py の rough_microdisk (L308-342) と完全同一の乱数シード (random.seed(0)) と
    インプレース移動平均フィルタ (Nsub=20, M=1000, half_s=1) で角度毎の半径 R(θ) を生成する。
    """
    n_pts = int(cir_reso)
    delta = float(dev_rate)
    n_sub = 20
    m_iter = 1000
    total_pts = n_pts * n_sub

    rng = random.Random(0)
    dev = np.array([delta * rng.uniform(-1.0, 1.0) for _ in range(total_pts)], dtype=np.float64)

    for _ in range(m_iter):
        for i in range(total_pts):
            dev[i] = 0.5 * (dev[i - 1] + dev[i])

    r_vals = (disk_d / 2.0) * (1.0 + dev[(np.arange(n_pts) * n_sub) % total_pts])
    return r_vals.astype(np.float32)


def get_structure_prefix(cfg: Dict[str, Any]) -> str:
    """spv/my_geometry.py と互換の構造プレフィックス文字列を返す。"""
    ncfg = normalize_cfg(cfg)
    stype = int(ncfg["structure_type"])
    period = float(ncfg["period"])
    n = float(ncfg["n_core"])
    g = ncfg["geom"]

    if stype == 6:
        r_ratio = float(g.get("r_ratio", 0.26))
        slab_ratio = float(g.get("slab_ratio", 0.50))
        n_ring = int(g.get("N_ring", 8))
        twist_deg = float(g.get("twist_deg", 3.5))
        core_r_ratio = float(g.get("core_r_ratio", 0.55))
        gap_ratio = float(g.get("gap_ratio", 0.02))
        bowtie_angle = float(g.get("bowtie_angle", 60.0))
        return (
            f"SBPhC({period:.3f} {n}, {r_ratio:.3f} {slab_ratio:.2f} "
            f"{n_ring} {twist_deg:.2f} {core_r_ratio:.2f} {gap_ratio:.4f} {bowtie_angle:.1f})"
        )
    elif stype == 3:
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        sac_d = float(g.get("Sac_D", 1.1))
        sac_t = float(g.get("Sac_t", 0.5))
        out_d = float(g.get("Out_D", 4.1))
        return f"md({disk_d:.2f} {disk_t:.3f}, {n}, {sac_d:.2f} {sac_t:.3f}, {out_d:.2f})"
    elif stype == 4:
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        sac_d = float(g.get("Sac_D", 1.1))
        sac_t = float(g.get("Sac_t", 0.5))
        out_d = float(g.get("Out_D", 4.1))
        cir_reso = int(g.get("Cir_reso", 180))
        dev_rate = float(g.get("Dev_rate", 0.02))
        return (
            f"rmd({disk_d:.2f} {disk_t:.3f}, {n}, {sac_d:.2f} {sac_t:.3f}, "
            f"{out_d:.2f}, {cir_reso} {dev_rate:.3f})"
        )
    elif stype == 5:
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        wg_w = float(g.get("wg_w", 0.4))
        wg_t = float(g.get("wg_t", 0.25))
        gap = float(g.get("gap", 0.1))
        out_d = float(g.get("Out_D", 4.1))
        return f"wg_d({disk_d:.2f} {disk_t:.3f}, {wg_w:.2f} {wg_t:.2f}, {gap:.2f}, {n}, {out_d:.2f})"
    elif stype == 0:
        r_ratio = float(g.get("r_ratio", 0.30))
        slab_ratio = float(g.get("slab_ratio", 0.60))
        nx = int(g.get("Nx", 12))
        ny = int(g.get("Ny", 10))
        def_n = int(g.get("defN", 3))
        return f"2D({period:.2f} {n}, {r_ratio:.3f} {slab_ratio:.2f} {nx} {ny} {def_n})"
    else:
        return f"{stype}_Custom_Structure"


def make_permittivity_point_fn(
    cfg: Dict[str, Any],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    指定された cfg に対し、任意の座標配列 (x, y, z) から比誘電率 ε(x, y, z) を返す
    純粋 JAX 関数を生成して返す。
    spv/my_geometry.py のオブジェクト上書き順序（後から配置したプリミティブが優先）を忠実に再現する。
    """
    ncfg = normalize_cfg(cfg)
    stype = int(ncfg["structure_type"])
    n_core = float(ncfg["n_core"])
    n_clad = float(ncfg.get("n_clad", 1.0))
    eps_core = n_core * n_core
    eps_clad = n_clad * n_clad

    period = float(ncfg["period"])
    g = ncfg["geom"]

    if stype == 6:
        # Singular Bowtie PhC (spv/my_geometry.py L411-508 と完全一致)
        r_ratio = float(g.get("r_ratio", 0.26))
        slab_ratio = float(g.get("slab_ratio", 0.50))
        n_ring = int(g.get("N_ring", 8))
        twist_deg = float(g.get("twist_deg", 3.5))
        core_r_ratio = float(g.get("core_r_ratio", 0.55))
        gap_ratio = float(g.get("gap_ratio", 0.02))
        bowtie_angle = float(g.get("bowtie_angle", 60.0))

        r_hole = r_ratio * period
        slab_t = round(slab_ratio * period, 3)
        core_r = core_r_ratio * period
        gap = gap_ratio * period
        half_gap = gap / 2.0
        half_apex = math.radians(bowtie_angle / 2.0)

        holes_np = _generate_twisted_bowtie_phc_holes(
            period=period,
            r=r_hole,
            n_ring=n_ring,
            twist_deg=twist_deg,
            core_r=core_r,
        )
        holes_xy = jnp.asarray(holes_np, dtype=jnp.float32)

        # 4. 誘電体ボウタイ・ナノアンテナ（x軸方向に対向する2つの三角形Prism: spv/my_geometry.py L477-493）
        x_outer = core_r * 1.1
        y_outer = (x_outer - half_gap) * math.tan(half_apex)

        v_right_0 = (half_gap, 0.0)
        v_right_1 = (x_outer, y_outer)
        v_right_2 = (x_outer, -y_outer)

        v_left_0 = (-half_gap, 0.0)
        v_left_1 = (-x_outer, -y_outer)
        v_left_2 = (-x_outer, y_outer)

        def _eps_fn_type6(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            # 1. ベーススラブ (|z| <= slab_t / 2)
            in_slab = sdf_slab_z(z, 0.0, slab_t) <= 0.0
            eps = jnp.where(in_slab, eps_core, eps_clad)
            # 2. ツイスト・ハニカム空気孔 (core_r + r < dist <= N_ring * period)
            if holes_xy.shape[0] > 0:
                in_holes = (sdf_multi_cylinders_xy(x, y, holes_xy, r_hole) <= 0.0) & in_slab
                eps = jnp.where(in_holes, eps_clad, eps)
            # 3. 中心欠陥部の円形空気領域 (半径 core_r)
            in_core_air = (jnp.hypot(x, y) <= core_r) & in_slab
            eps = jnp.where(in_core_air, eps_clad, eps)
            # 4. 左右2つの誘電体 Bowtie 三角柱 (x軸方向に対向) を上書き配置
            in_bt_right = sdf_extruded_triangle_z(x, y, z, v_right_0, v_right_1, v_right_2, 0.0, slab_t) <= 0.0
            in_bt_left = sdf_extruded_triangle_z(x, y, z, v_left_0, v_left_1, v_left_2, 0.0, slab_t) <= 0.0
            eps = jnp.where(in_bt_right | in_bt_left, eps_core, eps)
            return eps

        return _eps_fn_type6

    elif stype == 3:
        # Microdisk (spv/my_geometry.py L241-283 と完全一致)
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        sac_d = float(g.get("Sac_D", 1.1))
        sac_t = float(g.get("Sac_t", 0.5))
        out_d = float(g.get("Out_D", 4.1))
        sac_z = -(disk_t / 2.0 + sac_t / 2.0)
        sub_z = -(disk_t / 2.0 + sac_t + 1.5)
        etch_r = (out_d + disk_d - sac_d) / 2.0

        def _eps_fn_type3(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            r_xy = jnp.hypot(x, y)
            # 1. 全空間 air
            eps = jnp.full_like(x, eps_clad)
            # 2. スラブ Block(size=(inf, inf, disk_t), material=medium)
            eps = jnp.where(jnp.abs(z) <= 0.5 * disk_t, eps_core, eps)
            # 3. 外周空気円柱 Cylinder(Out_D / 2, height=inf, material=air)
            eps = jnp.where(r_xy <= 0.5 * out_d, eps_clad, eps)
            # 4. 下部基板 Block(center=(0,0,sub_z), size=(inf,inf,3), material=medium)
            eps = jnp.where(jnp.abs(z - sub_z) <= 1.5, eps_core, eps)
            # 5. ディスク本体 Cylinder(disk_D / 2, height=disk_t, material=medium)
            in_disk = (r_xy <= 0.5 * disk_d) & (jnp.abs(z) <= 0.5 * disk_t)
            eps = jnp.where(in_disk, eps_core, eps)
            # 6. 犠牲層 Block(center=(0,0,sac_z), size=(inf,inf,Sac_t), material=medium)
            in_sac_z = jnp.abs(z - sac_z) <= 0.5 * sac_t
            eps = jnp.where(in_sac_z, eps_core, eps)
            # 7. 犠牲層エッチング空気領域 Cylinder((Out_D + disk_D - Sac_D)/2, height=Sac_t, material=air)
            eps = jnp.where(in_sac_z & (r_xy <= etch_r), eps_clad, eps)
            # 8. 中央ペデスタル Cylinder(Sac_D / 2, height=Sac_t, material=medium)
            eps = jnp.where(in_sac_z & (r_xy <= 0.5 * sac_d), eps_core, eps)
            return eps

        return _eps_fn_type3

    elif stype == 4:
        # Rough Microdisk (spv/my_geometry.py L284-366 と完全一致)
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        sac_d = float(g.get("Sac_D", 1.1))
        sac_t = float(g.get("Sac_t", 0.5))
        out_d = float(g.get("Out_D", 4.1))
        cir_reso = int(g.get("Cir_reso", 180))
        dev_rate = float(g.get("Dev_rate", 0.02))
        sac_z = -(disk_t / 2.0 + sac_t / 2.0)
        sub_z = -(disk_t / 2.0 + sac_t + 1.5)
        etch_r = (out_d + disk_d - sac_d) / 2.0
        radii_j = jnp.asarray(_generate_rough_disk_radii(disk_d, cir_reso, dev_rate))

        def _eps_fn_type4(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            r_xy = jnp.hypot(x, y)
            eps = jnp.full_like(x, eps_clad)
            eps = jnp.where(jnp.abs(z) <= 0.5 * disk_t, eps_core, eps)
            eps = jnp.where(r_xy <= 0.5 * out_d, eps_clad, eps)
            eps = jnp.where(jnp.abs(z - sub_z) <= 1.5, eps_core, eps)
            in_rough_disk = sdf_extruded_star_polygon_z(x, y, z, radii_j, 0.0, disk_t) <= 0.0
            eps = jnp.where(in_rough_disk, eps_core, eps)
            in_sac_z = jnp.abs(z - sac_z) <= 0.5 * sac_t
            eps = jnp.where(in_sac_z, eps_core, eps)
            eps = jnp.where(in_sac_z & (r_xy <= etch_r), eps_clad, eps)
            eps = jnp.where(in_sac_z & (r_xy <= 0.5 * sac_d), eps_core, eps)
            return eps

        return _eps_fn_type4

    elif stype == 5:
        # Waveguide-coupled Microdisk (spv/my_geometry.py L367-410 と完全一致)
        disk_d = float(g.get("disk_D", 2.1))
        disk_t = float(g.get("disk_t", 0.25))
        wg_w = float(g.get("wg_w", 0.4))
        wg_t = float(g.get("wg_t", 0.25))
        gap = float(g.get("gap", 0.1))
        out_d = float(g.get("Out_D", 4.1))
        x_wg = 0.5 * disk_d + gap + 0.5 * wg_w

        def _eps_fn_type5(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            eps = jnp.full_like(x, eps_clad)
            # 1. 中央マイクロディスク Cylinder(disk_D / 2, height=disk_t, material=medium)
            in_disk = sdf_cylinder_z(x, y, z, 0.0, 0.0, 0.0, 0.5 * disk_d, disk_t) <= 0.0
            # 2. バス導波路 Block(center=((disk_D/2)+gap+(wg_w/2), 0, 0), size=(wg_w, inf, wg_t), material=medium)
            in_wg = sdf_box(x, y, z, x_wg, 0.0, 0.0, wg_w, 1e4, wg_t) <= 0.0
            eps = jnp.where(in_disk | in_wg, eps_core, eps)
            return eps

        return _eps_fn_type5

    else:
        # structure_type == 0: 2D PhC defect cavity (spv/my_geometry.py L79-116 と完全一致)
        if "r_ratio" in g:
            r_hole = float(g.get("r_ratio", 0.30)) * period
        elif "r" in g:
            r_hole = float(g["r"])
        else:
            r_hole = 0.30 * period

        if "slab_ratio" in g:
            slab_t = round(float(g.get("slab_ratio", 0.60)) * period, 3)
        elif "slab_t" in g:
            slab_t = float(g["slab_t"])
        else:
            slab_t = round(0.60 * period, 3)

        nx = int(g.get("Nx", g.get("Num_air", 12)))
        ny = int(g.get("Ny", g.get("Num_air", 10)))
        def_n = int(g.get("defN", 3))
        holes_xy = jnp.asarray(_generate_triangular_phc_holes(period, nx, ny, def_n))

        def _eps_fn_type0(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
            in_slab = sdf_slab_z(z, 0.0, slab_t) <= 0.0
            eps = jnp.where(in_slab, eps_core, eps_clad)
            if holes_xy.shape[0] > 0:
                in_holes = (sdf_multi_cylinders_xy(x, y, holes_xy, r_hole) <= 0.0) & in_slab
                eps = jnp.where(in_holes, eps_clad, eps)
            return eps

        return _eps_fn_type0


def build_structure_permittivity(
    cfg: Dict[str, Any],
    sp: Tuple[float, float, float],
    res: int,
    pz: int = 0,
    eps_averaging: bool = True,
    subpixel_n: int = 4,
    anisotropic: bool = False,
) -> Tuple[jnp.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    """
    指定されたセルサイズ sp = (Sx, Sy, Sz_full) と解像度 res [px/um] から
    3D 誘電率配列 ε(x, y, z) を GPU 上で構築して返す。
    """
    sx, sy, sz_full = float(sp[0]), float(sp[1]), float(sp[2])
    dx = 1.0 / float(res)

    nx = max(16, int(round(sx * res)))
    ny = max(16, int(round(sy * res)))
    nz_full = max(16, int(round(sz_full * res)))
    nx += nx % 2
    ny += ny % 2
    nz_full += nz_full % 2

    xs = (np.arange(nx, dtype=np.float32) - 0.5 * nx + 0.5) * dx
    ys = (np.arange(ny, dtype=np.float32) - 0.5 * ny + 0.5) * dx

    if pz != 0:
        # z >= 0 半領域: z_0 = +0.5 * dx からスタート (z=0 が半領域の下端境界 min_z に対応)
        nz = nz_full // 2
        zs = (np.arange(nz, dtype=np.float32) + 0.5) * dx
    else:
        nz = nz_full
        zs = (np.arange(nz, dtype=np.float32) - 0.5 * nz + 0.5) * dx

    eps_fn = make_permittivity_point_fn(cfg)
    eps_grid = evaluate_permittivity_grid(
        eps_point_fn=eps_fn,
        xs=xs,
        ys=ys,
        zs=zs,
        dx=dx,
        eps_averaging=eps_averaging,
        subpixel_n=subpixel_n,
        anisotropic=anisotropic,
    )
    prefix = get_structure_prefix(cfg)
    return eps_grid, (xs, ys, zs), prefix
