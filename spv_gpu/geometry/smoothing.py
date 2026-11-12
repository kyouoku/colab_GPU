"""
GPU-accelerated subpixel smoothing for permittivity arrays (Spec B1 §7).
Supports:
- Binary evaluation (eps_averaging=False) for STEP 1 coarse search
- 4x4x4 subpixel volume-fraction averaging (eps_averaging=True) for STEP 2.5 & STEP 3
- Optional Farjadpour-style harmonic/anisotropic averaging (anisotropic=True)
"""

from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np


def evaluate_permittivity_grid(
    eps_point_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    dx: float,
    eps_averaging: bool = True,
    subpixel_n: int = 4,
    anisotropic: bool = False,
    z_chunk_size: int = 16,
) -> jnp.ndarray:
    """
    3次元グリッド (xs, ys, zs) 上で誘電率分布 ε(x, y, z) を GPU 上で高速評価する。
    VRAM ピークを抑えるため z 軸方向にチャンク分割して `jax.jit` 評価する。

    Parameters
    ----------
    eps_point_fn : callable(x, y, z) -> eps
        任意座標配列 (x, y, z) に対して局所比誘電率 ε を返す JAX 関数。
    xs, ys, zs : np.ndarray
        1D 座標配列 (それぞれ長さ Nx, Ny, Nz)。
    dx : float
        空間グリッド間隔 [um]。
    eps_averaging : bool
        True の場合 subpixel_n^3 サブグリッド平均を行う。False の場合はセル中心のみ評価。
    subpixel_n : int
        1軸あたりのサブピクセル分割数（既定 4 -> 4x4x4 = 64点）。
    anisotropic : bool
        True の場合 (3, Nx, Ny, Nz) の対角テンソル（算術平均と調和平均の折衷）を返す。
        False の場合 (1, Nx, Ny, Nz) の等方性体積分率平均を返す。
    z_chunk_size : int
        1回に処理する z スライス数。

    Returns
    -------
    eps_grid : jnp.ndarray
        shape (1, Nx, Ny, Nz) または (3, Nx, Ny, Nz), dtype=float32
    """
    xs_j = jnp.asarray(xs, dtype=jnp.float32)
    ys_j = jnp.asarray(ys, dtype=jnp.float32)
    zs_j = jnp.asarray(zs, dtype=jnp.float32)

    if not eps_averaging or subpixel_n <= 1:
        @jax.jit
        def _eval_slice_binary(z_slice: jnp.ndarray) -> jnp.ndarray:
            xx, yy, zz = jnp.meshgrid(xs_j, ys_j, z_slice, indexing="ij")
            eps_val = eps_point_fn(xx, yy, zz).astype(jnp.float32)
            return eps_val[None, ...]  # (1, Nx, Ny, Nz_chunk)

        chunks = []
        nz = len(zs)
        for i0 in range(0, nz, z_chunk_size):
            z_sub = zs_j[i0 : i0 + z_chunk_size]
            chunks.append(_eval_slice_binary(z_sub))
        eps_out = jnp.concatenate(chunks, axis=-1)
        if anisotropic:
            eps_out = jnp.repeat(eps_out, 3, axis=0)
        return eps_out

    # サブピクセルオフセット: (-0.5 + (k + 0.5)/N) * dx
    offsets_1d = (jnp.arange(subpixel_n, dtype=jnp.float32) + 0.5) / float(subpixel_n) - 0.5
    offsets_1d = offsets_1d * float(dx)
    ox, oy, oz = jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    sub_offsets = jnp.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=-1)  # (N^3, 3)

    @jax.jit
    def _eval_slice_subpixel(z_slice: jnp.ndarray) -> jnp.ndarray:
        xx, yy, zz = jnp.meshgrid(xs_j, ys_j, z_slice, indexing="ij")

        def _eval_one_offset(off: jnp.ndarray):
            val = eps_point_fn(xx + off[0], yy + off[1], zz + off[2]).astype(jnp.float32)
            return val, 1.0 / val

        eps_vals, inv_eps_vals = jax.vmap(_eval_one_offset)(sub_offsets)
        eps_arith = jnp.mean(eps_vals, axis=0)  # <eps>
        inv_eps_mean = jnp.mean(inv_eps_vals, axis=0)  # <1/eps>
        eps_harm = 1.0 / jnp.maximum(inv_eps_mean, 1e-12)

        if anisotropic:
            # 簡易異方性平均: 面内(x,y)は算術平均寄り、法線変化の大きい成分に調和平均をブレンド
            eps_xy = 0.75 * eps_arith + 0.25 * eps_harm
            eps_z = 0.50 * eps_arith + 0.50 * eps_harm
            return jnp.stack([eps_xy, eps_xy, eps_z], axis=0)
        else:
            return eps_arith[None, ...]

    chunks = []
    nz = len(zs)
    # 4x4x4 展開時は1チャンクあたりのzスライス数を抑えてVRAM消費を最小化
    eff_chunk = max(1, z_chunk_size // 4)
    for i0 in range(0, nz, eff_chunk):
        z_sub = zs_j[i0 : i0 + eff_chunk]
        chunks.append(_eval_slice_subpixel(z_sub))

    return jnp.concatenate(chunks, axis=-1)
