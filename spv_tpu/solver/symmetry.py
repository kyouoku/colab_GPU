"""
Mirror symmetry handling for spv_tpu (TPU-optimized (Nz, Ny, Nx) Yee grid).
Fixes the Yee staggered-grid PEC/PMC mirror plane coupling:
- On the Yee grid, (Ex, Ey, Hz) reside at z = k*dx (mirror plane z=0 at k=0)
  while (Ez, Hx, Hy) reside at z = (k+0.5)*dx (ghost mirror at k=-1 -> z=-0.5*dx).
- PEC (-1, odd): Ex[0]=0, Ey[0]=0, Hz[0]=0 (and H_tan[-1] = +H_tan[0], so bwd_diff = 0 at k=0).
- PMC (+1, even): Do NOT zero Hx[0], Hy[0], Ez[0]! Mirror symmetry H_tan[-1] = -H_tan[0]
  is enforced via diff_bwd[0] = 2 * H_tan[0] inside the Yee stencil.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import jax.numpy as jnp
import numpy as np


COMP_TO_IDX: Dict[str, Tuple[str, int]] = {
    "Ex": ("E", 0),
    "Ey": ("E", 1),
    "Ez": ("E", 2),
    "Hx": ("H", 0),
    "Hy": ("H", 1),
    "Hz": ("H", 2),
}


def parity_label(pz: int) -> str:
    """パリティ値 (+1, -1, 0) を人間可読なラベル文字列に変換する。"""
    if pz == 1:
        return "even_TE_pz+1"
    elif pz == -1:
        return "odd_TM_pz-1"
    else:
        return "nosym_pz0"


def build_fdtdx_boundary_override(px: int = 0, py: int = 0, pz: int = 0) -> Dict[str, str]:
    """
    FDTDX の BoundaryConfig.from_uniform_bound に渡す override_types 辞書を構築する。
    +1 (even / TE-like) -> PMC, -1 (odd / TM-like) -> PEC, 0 -> PML.
    """
    overrides: Dict[str, str] = {}
    if px == 1:
        overrides["min_x"] = "pmc"
    elif px == -1:
        overrides["min_x"] = "pec"

    if py == 1:
        overrides["min_y"] = "pmc"
    elif py == -1:
        overrides["min_y"] = "pec"

    if pz == 1:
        overrides["min_z"] = "pmc"
    elif pz == -1:
        overrides["min_z"] = "pec"

    return overrides


def apply_e_mirror_bc_zyx(
    Ex: jnp.ndarray,
    Ey: jnp.ndarray,
    Ez: jnp.ndarray,
    px: int = 0,
    py: int = 0,
    pz: int = 0,
    is_y_first_shard: bool | jnp.ndarray = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    電場 (Ex, Ey, Ez) 各 (Nz, Ny, Nx) に対して PEC (-1) 鏡像対称面 (idx=0) を強制する。
    PMC (+1) では Yee 半格子成分をゼロ化せず、diff_bwd 内の 2*H[0] ゴースト条件で厳密処理する。
    """
    if pz == -1:
        Ex = Ex.at[0, :, :].set(0.0)
        Ey = Ey.at[0, :, :].set(0.0)

    if px == -1:
        Ey = Ey.at[:, :, 0].set(0.0)
        Ez = Ez.at[:, :, 0].set(0.0)

    if py == -1:
        if isinstance(is_y_first_shard, bool):
            if is_y_first_shard:
                Ex = Ex.at[:, 0, :].set(0.0)
                Ez = Ez.at[:, 0, :].set(0.0)
        else:
            Ex = Ex.at[:, 0, :].set(jnp.where(is_y_first_shard, 0.0, Ex[:, 0, :]))
            Ez = Ez.at[:, 0, :].set(jnp.where(is_y_first_shard, 0.0, Ez[:, 0, :]))

    return Ex, Ey, Ez


def apply_h_mirror_bc_zyx(
    Hx: jnp.ndarray,
    Hy: jnp.ndarray,
    Hz: jnp.ndarray,
    px: int = 0,
    py: int = 0,
    pz: int = 0,
    is_y_first_shard: bool | jnp.ndarray = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    磁場 (Hx, Hy, Hz) 各 (Nz, Ny, Nx) に対して PEC (-1) 鏡像対称面上の法線磁場 H_norm=0 を強制する。
    PMC (+1) では接線磁場が z=+0.5*dz に位置するため idx=0 をゼロ化しない。
    """
    if pz == -1:
        Hz = Hz.at[0, :, :].set(0.0)

    if px == -1:
        Hx = Hx.at[:, :, 0].set(0.0)

    if py == -1:
        if isinstance(is_y_first_shard, bool):
            if is_y_first_shard:
                Hy = Hy.at[:, 0, :].set(0.0)
        else:
            Hy = Hy.at[:, 0, :].set(jnp.where(is_y_first_shard, 0.0, Hy[:, 0, :]))

    return Hx, Hy, Hz


def apply_e_mirror_bc(E: jnp.ndarray, px: int = 0, py: int = 0, pz: int = 0) -> jnp.ndarray:
    """後方互換用: shape (3, Nx, Ny, Nz) の電場 E に対する PEC 鏡像対称面境界適用。"""
    if pz == -1:
        E = E.at[0, :, :, 0].set(0.0)
        E = E.at[1, :, :, 0].set(0.0)
    if px == -1:
        E = E.at[1, 0, :, :].set(0.0)
        E = E.at[2, 0, :, :].set(0.0)
    if py == -1:
        E = E.at[0, :, 0, :].set(0.0)
        E = E.at[2, :, 0, :].set(0.0)
    return E


def apply_h_mirror_bc(H: jnp.ndarray, px: int = 0, py: int = 0, pz: int = 0) -> jnp.ndarray:
    """後方互換用: shape (3, Nx, Ny, Nz) の磁場 H に対する PEC 鏡像対称面境界適用。"""
    if pz == -1:
        H = H.at[2, :, :, 0].set(0.0)
    if px == -1:
        H = H.at[0, 0, :, :].set(0.0)
    if py == -1:
        H = H.at[1, :, 0, :].set(0.0)
    return H


def expand_sources_for_full_domain_symmetry(
    src_positions: List[Tuple[float, float, float]],
    comp_name: str,
    pz: int,
) -> List[Tuple[Tuple[float, float, float], float]]:
    """
    フル領域計算時に z=0 対称でペアのダイポール源 (x, y, +z0) と (x, y, -z0) を
    同相 (+1.0) または逆相 (-1.0) で同時励振するためのリストを返す。
    """
    if pz == 0:
        return [(pos, 1.0) for pos in src_positions]

    even_positive_comps = {"Ex", "Ey", "Hz"}
    base_sign = 1.0 if comp_name in even_positive_comps else -1.0
    mirror_sign = float(pz) * base_sign

    expanded: List[Tuple[Tuple[float, float, float], float]] = []
    for x, y, z in src_positions:
        if abs(z) < 1e-9:
            if mirror_sign > 0:
                expanded.append(((x, y, 0.0), 1.0))
        else:
            expanded.append(((x, y, abs(z)), 0.5))
            expanded.append(((x, y, -abs(z)), 0.5 * mirror_sign))
    return expanded


def unfold_2d_slice_z(
    slice_half: np.ndarray,
    zs_half: np.ndarray,
    pz: int,
    comp_name: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    半領域 (z >= 0) で記録された xz または yz 断面データ (shape: (N_horiz, Nz_half)) を
    フル領域 (-z_max ~ +z_max) に鏡像展開する。
    zs_half[0] == 0.0 を含む場合は z=0 列を重複させずに展開する。
    """
    if pz == 0:
        return slice_half, zs_half

    if comp_name is None:
        sign = 1.0
    else:
        even_positive_comps = {"Ex", "Ey", "Hz"}
        base_sign = 1.0 if comp_name in even_positive_comps else -1.0
        sign = float(pz) * base_sign

    if len(zs_half) > 1 and abs(float(zs_half[0])) < 1e-9:
        neg_part = sign * slice_half[:, 1:][:, ::-1]
        full_slice = np.concatenate([neg_part, slice_half], axis=-1)
        zs_full = np.concatenate([-zs_half[1:][::-1], zs_half], axis=0)
    else:
        neg_part = sign * slice_half[:, ::-1]
        full_slice = np.concatenate([neg_part, slice_half], axis=-1)
        zs_full = np.concatenate([-zs_half[::-1], zs_half], axis=0)
    return full_slice, zs_full
