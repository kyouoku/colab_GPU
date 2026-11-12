"""
GPU-accelerated subpixel smoothing for permittivity arrays (Spec B1 §7).
Supports:
- Binary evaluation (eps_averaging=False) for STEP 1 coarse search
- 4x4x4 subpixel volume-fraction averaging (eps_averaging=True) for STEP 2.5 & STEP 3
- Optional Farjadpour-style harmonic/anisotropic averaging (anisotropic=True)
- Streams z-chunks from GPU to host NumPy array so the full 3D eps_grid consumes 0 GPU VRAM at rest.
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
    z_chunk_size: int = 8,
) -> np.ndarray:
    """
    3次元グリッド (xs, ys, zs) 上で誘電率分布 ε(x, y, z) を GPU 上で高速評価する。
    VRAM ピークおよび常駐メモリを最小化するため、z 軸方向に小チャンク分割して GPU で評価し、
    即座にホスト (CPU) の NumPy 配列へ転送して返す。

    Returns
    -------
    eps_grid : np.ndarray
        shape (1, Nx, Ny, Nz) または (3, Nx, Ny, Nz), dtype=float32 (CPU NumPy array)
    """
    xs_j = jnp.asarray(xs, dtype=jnp.float32)
    ys_j = jnp.asarray(ys, dtype=jnp.float32)
    zs_j = jnp.asarray(zs, dtype=jnp.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    n_comp = 3 if anisotropic else 1
    eps_out = np.empty((n_comp, nx, ny, nz), dtype=np.float32)

    if not eps_averaging or subpixel_n <= 1:
        @jax.jit
        def _eval_slice_binary(z_slice: jnp.ndarray) -> jnp.ndarray:
            xx, yy, zz = jnp.meshgrid(xs_j, ys_j, z_slice, indexing="ij")
            eps_val = eps_point_fn(xx, yy, zz).astype(jnp.float32)
            return eps_val[None, ...]  # (1, Nx, Ny, Nz_chunk)

        for i0 in range(0, nz, z_chunk_size):
            i1 = min(i0 + z_chunk_size, nz)
            z_sub = zs_j[i0:i1]
            chunk_np = np.asarray(_eval_slice_binary(z_sub), dtype=np.float32)
            if anisotropic:
                eps_out[:, :, :, i0:i1] = np.repeat(chunk_np, 3, axis=0)
            else:
                eps_out[:, :, :, i0:i1] = chunk_np
        return eps_out

    # サブピクセルオフセット: (-0.5 + (k + 0.5)/N) * dx
    offsets_1d = (jnp.arange(subpixel_n, dtype=jnp.float32) + 0.5) / float(subpixel_n) - 0.5
    offsets_1d = offsets_1d * float(dx)
    ox, oy, oz = jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    sub_offsets = jnp.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=-1)  # (N^3, 3)

    @jax.jit
    def _eval_slice_subpixel(z_slice: jnp.ndarray) -> jnp.ndarray:
        xx, yy, zz = jnp.meshgrid(xs_j, ys_j, z_slice, indexing="ij")

        def _carry_fn(acc, off: jnp.ndarray):
            sum_eps, sum_inv_eps = acc
            val = eps_point_fn(xx + off[0], yy + off[1], zz + off[2]).astype(jnp.float32)
            return (sum_eps + val, sum_inv_eps + 1.0 / val), None

        init_acc = (
            jnp.zeros_like(xx, dtype=jnp.float32),
            jnp.zeros_like(xx, dtype=jnp.float32),
        )
        (total_eps, total_inv_eps), _ = jax.lax.scan(_carry_fn, init_acc, sub_offsets)
        n_pts = float(sub_offsets.shape[0])
        eps_arith = total_eps / n_pts
        inv_eps_mean = total_inv_eps / n_pts
        eps_harm = 1.0 / jnp.maximum(inv_eps_mean, 1e-12)

        if anisotropic:
            eps_xy = 0.75 * eps_arith + 0.25 * eps_harm
            eps_z = 0.50 * eps_arith + 0.50 * eps_harm
            return jnp.stack([eps_xy, eps_xy, eps_z], axis=0)
        else:
            return eps_arith[None, ...]

    eff_chunk = max(1, z_chunk_size // 2)
    for i0 in range(0, nz, eff_chunk):
        i1 = min(i0 + eff_chunk, nz)
        z_sub = zs_j[i0:i1]
        eps_out[:, :, :, i0:i1] = np.asarray(_eval_slice_subpixel(z_sub), dtype=np.float32)

    return eps_out
