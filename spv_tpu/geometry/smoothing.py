"""
TPU-accelerated subpixel smoothing for permittivity arrays.
Evaluates directly in (Nz_chunk, Ny, Nx) layout using (1, Ny, Nx) + (Nz_chunk, 1, 1) broadcasting
and fixed-size z-chunk padding so XLA compiles only once per grid.
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
    layout: str = "xyz",
) -> np.ndarray:
    """
    3次元グリッド (xs, ys, zs) 上で誘電率分布 ε(x, y, z) を TPU/JAX 上で高速評価する。
    (Nz_chunk, Ny, Nx) で計算してホスト NumPy 配列へストリーミング転送し、
    最終チャンクを固定長にパディングすることで末尾での XLA 再コンパイルを回避する。

    Parameters
    ----------
    layout : str
        "zyx" -> shape (n_comp, Nz, Ny, Nx) (TPU カーネル向けゼロコピー)
        "xyz" -> shape (n_comp, Nx, Ny, Nz) (後方互換)
    """
    xs_j = jnp.asarray(xs, dtype=jnp.float32)
    ys_j = jnp.asarray(ys, dtype=jnp.float32)
    zs_j = jnp.asarray(zs, dtype=jnp.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    n_comp = 3 if anisotropic else 1
    layout_l = str(layout).lower()
    if layout_l not in ("zyx", "xyz"):
        raise ValueError(f"Unsupported layout: {layout!r}. Expected 'zyx' or 'xyz'.")

    if layout_l == "zyx":
        eps_out = np.empty((n_comp, nz, ny, nx), dtype=np.float32)
    else:
        eps_out = np.empty((n_comp, nx, ny, nz), dtype=np.float32)

    if not eps_averaging or subpixel_n <= 1:
        eff_chunk = max(1, min(int(z_chunk_size), nz))

        @jax.jit
        def _eval_slice_binary(z_slice: jnp.ndarray) -> jnp.ndarray:
            nz_c = z_slice.shape[0]
            xx = jnp.broadcast_to(xs_j[None, None, :], (1, ny, nx))
            yy = jnp.broadcast_to(ys_j[None, :, None], (1, ny, nx))
            zz = z_slice[:, None, None]
            eps_val = jnp.broadcast_to(
                eps_point_fn(xx, yy, zz).astype(jnp.float32),
                (nz_c, ny, nx),
            )
            return eps_val[None, ...]  # (1, Nz_chunk, Ny, Nx)

        for i0 in range(0, nz, eff_chunk):
            i1 = min(i0 + eff_chunk, nz)
            valid_len = i1 - i0
            z_sub = zs_j[i0:i1]
            if valid_len < eff_chunk:
                pad_n = eff_chunk - valid_len
                z_sub = jnp.concatenate([z_sub, jnp.repeat(z_sub[-1:], pad_n, axis=0)], axis=0)
            chunk_np = np.asarray(_eval_slice_binary(z_sub), dtype=np.float32)[:, :valid_len, :, :]
            if anisotropic:
                chunk_np = np.repeat(chunk_np, 3, axis=0)
            if layout_l == "zyx":
                eps_out[:, i0:i1, :, :] = chunk_np
            else:
                eps_out[:, :, :, i0:i1] = np.transpose(chunk_np, (0, 3, 2, 1))
        return eps_out

    offsets_1d = (jnp.arange(subpixel_n, dtype=jnp.float32) + 0.5) / float(subpixel_n) - 0.5
    offsets_1d = offsets_1d * float(dx)
    ox, oy, oz = jnp.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    sub_offsets = jnp.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=-1)  # (N^3, 3)

    eff_chunk = max(1, min(8, int(z_chunk_size) // 2, nz))

    @jax.jit
    def _eval_slice_subpixel(z_slice: jnp.ndarray) -> jnp.ndarray:
        nz_c = z_slice.shape[0]
        xx = jnp.broadcast_to(xs_j[None, None, :], (1, ny, nx))
        yy = jnp.broadcast_to(ys_j[None, :, None], (1, ny, nx))
        zz = z_slice[:, None, None]

        def _carry_fn(acc, off: jnp.ndarray):
            sum_eps, sum_inv_eps = acc
            val = jnp.broadcast_to(
                eps_point_fn(xx + off[0], yy + off[1], zz + off[2]).astype(jnp.float32),
                (nz_c, ny, nx),
            )
            return (sum_eps + val, sum_inv_eps + 1.0 / val), None

        init_acc = (
            jnp.zeros((nz_c, ny, nx), dtype=jnp.float32),
            jnp.zeros((nz_c, ny, nx), dtype=jnp.float32),
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
        return eps_arith[None, ...]

    for i0 in range(0, nz, eff_chunk):
        i1 = min(i0 + eff_chunk, nz)
        valid_len = i1 - i0
        z_sub = zs_j[i0:i1]
        if valid_len < eff_chunk:
            pad_n = eff_chunk - valid_len
            z_sub = jnp.concatenate([z_sub, jnp.repeat(z_sub[-1:], pad_n, axis=0)], axis=0)
        chunk_np = np.asarray(_eval_slice_subpixel(z_sub), dtype=np.float32)[:, :valid_len, :, :]
        if layout_l == "zyx":
            eps_out[:, i0:i1, :, :] = chunk_np
        else:
            eps_out[:, :, :, i0:i1] = np.transpose(chunk_np, (0, 3, 2, 1))

    return eps_out
