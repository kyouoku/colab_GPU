"""
JAX-compatible Signed Distance Functions (SDF) for 2D/3D photonic structures (Spec B1 §7).
Convention: SDF(x, y, z) < 0 inside the shape, > 0 outside the shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sdf_slab_z(z: jnp.ndarray, z_center: float, thickness: float) -> jnp.ndarray:
    """z方向に有限厚み thickness を持つ無限スラブの SDF。"""
    return jnp.abs(z - z_center) - 0.5 * thickness


def sdf_cylinder_z(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    height: float,
) -> jnp.ndarray:
    """z軸平行円柱の 3D SDF。"""
    d_radial = jnp.hypot(x - cx, y - cy) - radius
    d_axial = jnp.abs(z - cz) - 0.5 * height
    outside = jnp.hypot(jnp.maximum(d_radial, 0.0), jnp.maximum(d_axial, 0.0))
    inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
    return outside + inside


def sdf_multi_cylinders_xy(
    x: jnp.ndarray,
    y: jnp.ndarray,
    centers_xy: jnp.ndarray,
    radius: float,
) -> jnp.ndarray:
    """
    複数のz軸平行円柱群（同一半径 radius）の 2D 面内 SDF の最小値（和集合）。
    centers_xy: shape (M, 2)
    x, y: 任意形状の配列
    """
    if centers_xy.shape[0] == 0:
        return jnp.full_like(x, 1e6)
    dx = x[..., None] - centers_xy[:, 0]
    dy = y[..., None] - centers_xy[:, 1]
    dist_sq = jnp.min(dx * dx + dy * dy, axis=-1)
    return jnp.sqrt(dist_sq) - radius


def sdf_box(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
) -> jnp.ndarray:
    """直方体 (Block) の 3D SDF。"""
    qx = jnp.abs(x - cx) - 0.5 * sx
    qy = jnp.abs(y - cy) - 0.5 * sy
    qz = jnp.abs(z - cz) - 0.5 * sz
    outside = jnp.sqrt(
        jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2 + jnp.maximum(qz, 0.0) ** 2
    )
    inside = jnp.minimum(jnp.maximum(qx, jnp.maximum(qy, qz)), 0.0)
    return outside + inside


def sdf_triangle_2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    v0: tuple[float, float],
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> jnp.ndarray:
    """
    2D 三角形 (v0, v1, v2) の厳密な符号付き距離関数 (Inigo Quilez triangle SDF)。
    内部で負 (< 0)、外部で正 (> 0)。
    """
    p0x, p0y = float(v0[0]), float(v0[1])
    p1x, p1y = float(v1[0]), float(v1[1])
    p2x, p2y = float(v2[0]), float(v2[1])

    e0x, e0y = p1x - p0x, p1y - p0y
    e1x, e1y = p2x - p1x, p2y - p1y
    e2x, e2y = p0x - p2x, p0y - p2y

    v0x, v0y = x - p0x, y - p0y
    v1x, v1y = x - p1x, y - p1y
    v2x, v2y = x - p2x, y - p2y

    t0 = jnp.clip((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y + 1e-30), 0.0, 1.0)
    t1 = jnp.clip((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y + 1e-30), 0.0, 1.0)
    t2 = jnp.clip((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y + 1e-30), 0.0, 1.0)

    pq0x, pq0y = v0x - e0x * t0, v0y - e0y * t0
    pq1x, pq1y = v1x - e1x * t1, v1y - e1y * t1
    pq2x, pq2y = v2x - e2x * t2, v2y - e2y * t2

    d0 = pq0x * pq0x + pq0y * pq0y
    d1 = pq1x * pq1x + pq1y * pq1y
    d2 = pq2x * pq2x + pq2y * pq2y
    d_min = jnp.minimum(d0, jnp.minimum(d1, d2))

    # 巻き順 (winding orientation) の符号を正規化
    s = jnp.sign(e0x * e2y - e0y * e2x)
    c0 = s * (v0x * e0y - v0y * e0x)
    c1 = s * (v1x * e1y - v1y * e1x)
    c2 = s * (v2x * e2y - v2y * e2x)
    c_min = jnp.minimum(c0, jnp.minimum(c1, c2))

    sign = -jnp.sign(c_min)
    return jnp.sqrt(jnp.maximum(d_min, 0.0)) * jnp.where(sign == 0.0, -1.0, sign)


def sdf_extruded_triangle_z(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    v0: tuple[float, float],
    v1: tuple[float, float],
    v2: tuple[float, float],
    cz: float,
    height: float,
) -> jnp.ndarray:
    """2D 三角形を z方向に高さ height で押し出した三角柱 (Prism) の 3D SDF。"""
    d_xy = sdf_triangle_2d(x, y, v0, v1, v2)
    d_z = jnp.abs(z - cz) - 0.5 * height
    outside = jnp.hypot(jnp.maximum(d_xy, 0.0), jnp.maximum(d_z, 0.0))
    inside = jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)
    return outside + inside


def sdf_extruded_polygon_z(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    vertices_xy: jnp.ndarray,
    cz: float,
    height: float,
) -> jnp.ndarray:
    """
    任意の2D多角形 (vertices_xy: shape (N, 2)) を z方向に押し出した Prism の 3D SDF。
    """
    vx = vertices_xy[:, 0]
    vy = vertices_xy[:, 1]
    vx_next = jnp.roll(vx, -1)
    vy_next = jnp.roll(vy, -1)

    ex = vx_next - vx
    ey = vy_next - vy
    elen_sq = ex * ex + ey * ey + 1e-30

    px = x[..., None] - vx
    py = y[..., None] - vy

    t = jnp.clip((px * ex + py * ey) / elen_sq, 0.0, 1.0)
    bx = px - ex * t
    by = py - ey * t
    d_sq = jnp.min(bx * bx + by * by, axis=-1)

    # Winding number (ray-casting parity) for inside/outside
    cond1 = vy <= y[..., None]
    cond2 = vy_next > y[..., None]
    cond3 = (ex * py - ey * px) > 0.0
    up_cross = cond1 & cond2 & cond3
    down_cross = (~cond1) & (~cond2) & (~cond3)
    wn = jnp.sum(up_cross.astype(jnp.int32) - down_cross.astype(jnp.int32), axis=-1)
    sign = jnp.where(wn != 0, -1.0, 1.0)

    d_xy = sign * jnp.sqrt(jnp.maximum(d_sq, 0.0))
    d_z = jnp.abs(z - cz) - 0.5 * height
    outside = jnp.hypot(jnp.maximum(d_xy, 0.0), jnp.maximum(d_z, 0.0))
    inside = jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)
    return outside + inside


def sdf_extruded_star_polygon_z(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    radii_at_angles: jnp.ndarray,
    cz: float,
    height: float,
) -> jnp.ndarray:
    """
    角度 θ ∈ [0, 2π) ごとに半径 R(θ) が定義されたラフネス付きマイクロディスク (type 4) の高速 3D SDF。
    radii_at_angles: shape (Cir_reso,) で θ_i = 2π * i / Cir_reso における半径配列。
    """
    n_verts = radii_at_angles.shape[0]
    theta = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)
    f_idx = theta / (2.0 * jnp.pi) * n_verts
    idx0 = jnp.floor(f_idx).astype(jnp.int32) % n_verts
    idx1 = (idx0 + 1) % n_verts
    frac = f_idx - jnp.floor(f_idx)

    # 頂点間の線形補間 (多角形近似と一致)
    r_local = (1.0 - frac) * radii_at_angles[idx0] + frac * radii_at_angles[idx1]
    d_xy = jnp.hypot(x, y) - r_local
    d_z = jnp.abs(z - cz) - 0.5 * height
    outside = jnp.hypot(jnp.maximum(d_xy, 0.0), jnp.maximum(d_z, 0.0))
    inside = jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)
    return outside + inside


def sdf_union(d1: jnp.ndarray, d2: jnp.ndarray) -> jnp.ndarray:
    """CSG 和集合 (A ∪ B)。"""
    return jnp.minimum(d1, d2)


def sdf_intersection(d1: jnp.ndarray, d2: jnp.ndarray) -> jnp.ndarray:
    """CSG 積集合 (A ∩ B)。"""
    return jnp.maximum(d1, d2)


def sdf_difference(d1: jnp.ndarray, d2: jnp.ndarray) -> jnp.ndarray:
    """CSG 差集合 (A \\ B): d1 から d2 をくり抜く。"""
    return jnp.maximum(d1, -d2)
