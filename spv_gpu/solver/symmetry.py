"""
Mirror symmetry handling for FDTDX (Spec B1 §6).
Supports:
1. Half-domain reduction (z >= 0, and optionally x >= 0, y >= 0) with PEC (-1) / PMC (+1)
   boundary condition at the mirror plane (min_z / min_x / min_y).
2. Explicit tangential field zeroing masks on the Yee grid at z = 0 (and x = 0, y = 0).
3. Full-domain symmetric source placement (fallback when half-domain reduction is disabled).
4. Unfolding of half-domain 2D cross-sections (xz, yz) back to full domain for visualization.
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
    +1 (even / TE-like) -> PMC (Perfect Magnetic Conductor: H_tan = 0, E_tan != 0)
    -1 (odd  / TM-like) -> PEC (Perfect Electric Conductor: E_tan = 0, H_tan != 0)
     0 (対称性なし)     -> PML
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


def apply_e_mirror_bc(E: jnp.ndarray, px: int = 0, py: int = 0, pz: int = 0) -> jnp.ndarray:
    """
    電場 E (shape: (3, Nx, Ny, Nz)) に対して鏡像対称面 (idx=0) の境界条件を強制する。
    - PEC (-1, odd): 接線電場 E_tan = 0
    - PMC (+1, even): 法線電場 E_norm = 0
    """
    if pz == -1:
        # z=0 面が PEC (TM-like odd): Ex, Ey = 0 at z_idx=0
        E = E.at[0, :, :, 0].set(0.0)
        E = E.at[1, :, :, 0].set(0.0)
    elif pz == 1:
        # z=0 面が PMC (TE-like even): Ez = 0 at z_idx=0
        E = E.at[2, :, :, 0].set(0.0)

    if px == -1:
        E = E.at[1, 0, :, :].set(0.0)
        E = E.at[2, 0, :, :].set(0.0)
    elif px == 1:
        E = E.at[0, 0, :, :].set(0.0)

    if py == -1:
        E = E.at[0, :, 0, :].set(0.0)
        E = E.at[2, :, 0, :].set(0.0)
    elif py == 1:
        E = E.at[1, :, 0, :].set(0.0)

    return E


def apply_h_mirror_bc(H: jnp.ndarray, px: int = 0, py: int = 0, pz: int = 0) -> jnp.ndarray:
    """
    磁場 H (shape: (3, Nx, Ny, Nz)) に対して鏡像対称面 (idx=0) の境界条件を強制する。
    - PMC (+1, even): 接線磁場 H_tan = 0
    - PEC (-1, odd): 法線磁場 H_norm = 0
    """
    if pz == 1:
        # z=0 面が PMC (TE-like even): Hx, Hy = 0 at z_idx=0
        H = H.at[0, :, :, 0].set(0.0)
        H = H.at[1, :, :, 0].set(0.0)
    elif pz == -1:
        # z=0 面が PEC (TM-like odd): Hz = 0 at z_idx=0
        H = H.at[2, :, :, 0].set(0.0)

    if px == 1:
        H = H.at[1, 0, :, :].set(0.0)
        H = H.at[2, 0, :, :].set(0.0)
    elif px == -1:
        H = H.at[0, 0, :, :].set(0.0)

    if py == 1:
        H = H.at[0, :, 0, :].set(0.0)
        H = H.at[2, :, 0, :].set(0.0)
    elif py == -1:
        H = H.at[1, :, 0, :].set(0.0)

    return H


def expand_sources_for_full_domain_symmetry(
    src_positions: List[Tuple[float, float, float]],
    comp_name: str,
    pz: int,
) -> List[Tuple[Tuple[float, float, float], float]]:
    """
    仕様書B1 §6 次善策（フォールバック）用:
    フル領域計算時に z=0 対称でペアのダイポール源 (x, y, +z0) と (x, y, -z0) を
    同相 (+1.0) または逆相 (-1.0) で同時励振するためのリストを返す。
    """
    if pz == 0:
        return [(pos, 1.0) for pos in src_positions]

    # z鏡像に対する各成分の偶奇符号 (pz = +1 のとき Ex, Ey, Hz が +1, Hx, Hy, Ez が -1)
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
    フル領域 (-z_max ~ +z_max, shape: (N_horiz, 2*Nz_half)) に鏡像展開する。
    エネルギー密度 (comp_name=None) の場合は常に偶対称 (+1) で展開する。
    """
    if pz == 0:
        return slice_half, zs_half

    if comp_name is None:
        sign = 1.0
    else:
        even_positive_comps = {"Ex", "Ey", "Hz"}
        base_sign = 1.0 if comp_name in even_positive_comps else -1.0
        sign = float(pz) * base_sign

    neg_part = sign * slice_half[:, ::-1]
    full_slice = np.concatenate([neg_part, slice_half], axis=-1)
    zs_full = np.concatenate([-zs_half[::-1], zs_half], axis=0)
    return full_slice, zs_full
