"""
TPU-optimized JAX 3D FDTD simulation runner for spv_tpu.
Key TPU optimizations over spv_gpu:
1. (Nz, Ny, Nx) memory layout aligned to TPU VPU (8 x 128) tiles so XLA never pads inner dimensions wastefully.
2. Zero jnp.roll stencils using explicit slice differences + exact Yee PEC/PMC mirror plane coupling.
3. Thin boundary-slab Roden-Gedney CFS-CPML (`pml_mode='cpml'`, default) reducing HBM usage ~3x vs FDTDX
   while providing strong evanescent/slab absorption (~ -60 to -80 dB) over scalar sponge damping.
4. Automatic multi-chip TPU y-axis spatial domain sharding via `shard_map` + 1-slice `lax.ppermute` halos.
5. Persistent XLA compilation cache + bounded FIFO JIT cache + array-argument parameter passing to eliminate
   recompilation between STEP 2.5 and STEP 3.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import gc
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
import jax
import jax.numpy as jnp
import numpy as np

from ..config import detect_tpu_hardware
from .symmetry import (
    COMP_TO_IDX,
    apply_e_mirror_bc,
    apply_e_mirror_bc_zyx,
    apply_h_mirror_bc,
    apply_h_mirror_bc_zyx,
    build_fdtdx_boundary_override,
    unfold_2d_slice_z,
)

try:
    from jax import shard_map as _jax_shard_map  # JAX >= 0.6
except Exception:
    try:
        from jax.experimental.shard_map import shard_map as _jax_shard_map
    except Exception:
        _jax_shard_map = None

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


_MAX_JIT_CACHE_ENTRIES = 4
_JIT_CHUNK_CACHE: "OrderedDict[Tuple[Any, ...], Callable]" = OrderedDict()
_XLA_CACHE_CONFIGURED_DIR: Optional[str] = None


def configure_tpu_xla_cache(cache_dir: Optional[str] = "/tmp/spv_tpu_xla_cache") -> Optional[str]:
    """XLA 永続コンパイルキャッシュを有効化し、同一構成での再コンパイル時間を劇的に短縮する。"""
    global _XLA_CACHE_CONFIGURED_DIR
    if not cache_dir:
        return None
    if _XLA_CACHE_CONFIGURED_DIR == cache_dir:
        return cache_dir
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        _XLA_CACHE_CONFIGURED_DIR = cache_dir
    except Exception:
        return None
    return cache_dir


def clear_jit_cache() -> None:
    """インメモリ JIT キャッシュおよび JAX/XLA のコンパイル済み実行グラフキャッシュを解放する。"""
    _JIT_CHUNK_CACHE.clear()
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def check_tpu_environment(
    nx: int,
    ny: int,
    nz: int,
    dtype: str = "float32",
    pml_mode: str = "cpml",
    require_tpu: bool = False,
) -> Dict[str, Any]:
    """
    JAX の利用可能デバイス (TPU/GPU/CPU)・推定 HBM 使用量・タイル整合性を事前チェックする。
    """
    hw = detect_tpu_hardware()
    platform = hw["platform"]
    device_kind = hw["device_kind"]
    num_devices = int(hw["num_devices"])

    if require_tpu and platform != "tpu":
        raise RuntimeError(
            f"TPU が検出されませんでした (現在のバックエンド: {platform})。"
            "Google Colab の「ランタイム > ランタイムのタイプを変更」から TPU (v5e-1 / v6e-1 / v2-8 等) を選択してください。"
        )

    from ..config import estimate_tpu_memory_gb

    est_gb = estimate_tpu_memory_gb(nx=nx, ny=ny, nz=nz, dtype=dtype, pml_mode=pml_mode)
    return {
        "platform": platform,
        "device_kind": device_kind,
        "num_devices": num_devices,
        "per_device_hbm_gb": hw["per_device_gb"],
        "total_hbm_gb": hw["total_hbm_gb"],
        "has_fdtdx": False,
        "fdtdx_version": f"tpu_native_{pml_mode}",
        "grid_shape": (nx, ny, nz),
        "total_cells": nx * ny * nz,
        "estimated_hbm_gb": est_gb,
        "estimated_vram_gb": est_gb,
    }


def check_gpu_environment(
    nx: int,
    ny: int,
    nz: int,
    dtype: str = "float32",
    require_gpu: bool = False,
) -> Dict[str, Any]:
    """後方互換エイリアス (`check_tpu_environment` を呼ぶ)。"""
    info = check_tpu_environment(nx=nx, ny=ny, nz=nz, dtype=dtype, require_tpu=False)
    if require_gpu and info["platform"] not in ("gpu", "tpu"):
        raise RuntimeError(f"アクセラレータが検出されませんでした ({info['platform']})。")
    return info


@dataclass
class TPUSimulationResult:
    """1回の TPU FDTD 実行結果を保持するデータクラス。"""

    time_array: np.ndarray
    monitor_traces: np.ndarray
    monitor_positions: List[Tuple[float, float, float]]
    monitor_components: List[str]
    dt_meep: float
    dt_rec: float
    t_src_end: float
    total_steps_run: int
    stopped_early_by_decay: bool
    peak_energy: float
    final_energy: float
    wall_time_sec: float
    jit_compile_sec: float
    mcells_per_sec: float
    grid_shape: Tuple[int, int, int]
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray]
    pz: int
    dft_slices: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    dft_freqs: List[float] = field(default_factory=list)
    pml_mode: str = "cpml"
    n_shards: int = 1
    device_kind: str = "unknown"


FDTDXSimulationResult = TPUSimulationResult


def _coord_to_index(coord_1d: np.ndarray, val: float) -> int:
    """1D 座標配列から最も近いセルインデックスを返す。"""
    return int(np.clip(np.argmin(np.abs(coord_1d - val)), 0, len(coord_1d) - 1))


def _build_sponge_1d_profile(n: int, pml_cells: int, min_is_pml: bool = True, max_is_pml: bool = True) -> np.ndarray:
    """1D 3次多項式グレーディング減衰係数 exp(-σ(i)) を生成する (`pml_mode='sponge'` 用)。"""
    idx = np.arange(n, dtype=np.float32)
    sigma = np.zeros(n, dtype=np.float32)
    sigma_max = 0.8 * 4.0 / max(pml_cells, 1)
    if min_is_pml and pml_cells > 0:
        d_min = np.clip((pml_cells - idx) / float(pml_cells), 0.0, 1.0)
        sigma = np.maximum(sigma, sigma_max * (d_min**3))
    if max_is_pml and pml_cells > 0:
        d_max = np.clip((idx - (n - 1 - pml_cells)) / float(pml_cells), 0.0, 1.0)
        sigma = np.maximum(sigma, sigma_max * (d_max**3))
    return np.exp(-sigma).astype(np.float32)


def _build_cpml_1d_coeffs(
    n: int,
    pml_cells: int,
    dx: float,
    dt_meep: float,
    min_is_pml: bool,
    max_is_pml: bool,
    dtype_np: Any = np.float32,
) -> Tuple[np.ndarray, ...]:
    """
    Roden-Gedney CFS-CPML の 1D 係数配列 (a_min_e, b_min_e, a_max_e, b_max_e, ik_e,
    a_min_h, b_min_h, a_max_h, b_max_h, ik_h) を構築する。
    """
    p = max(1, int(pml_cells))
    m_poly = 3.0
    sigma_max = 0.8 * (m_poly + 1.0) / max(dx, 1e-12)
    kappa_max = 5.0
    alpha_max = 0.05 / max(dx, 1e-12)

    def _compute_abk(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.clip(d, 0.0, 1.0)
        sigma = sigma_max * (d**m_poly)
        kappa = 1.0 + (kappa_max - 1.0) * (d**m_poly)
        alpha = alpha_max * (1.0 - d)
        b = np.exp(-(sigma / kappa + alpha) * dt_meep)
        denom = sigma * kappa + (kappa * kappa) * alpha
        a = np.where(sigma > 0.0, (sigma / np.where(denom > 0.0, denom, 1.0)) * (b - 1.0), 0.0)
        return a.astype(dtype_np), b.astype(dtype_np), (1.0 / kappa).astype(dtype_np)

    # Min 側 (インデックス 0 .. p-1)
    idx_min = np.arange(p, dtype=np.float64)
    d_min_e = (p - idx_min) / float(p)
    d_min_h = (p - (idx_min + 0.5)) / float(p)
    if min_is_pml:
        a_min_e, b_min_e, ik_min_e = _compute_abk(d_min_e)
        a_min_h, b_min_h, ik_min_h = _compute_abk(d_min_h)
    else:
        a_min_e = np.zeros(p, dtype=dtype_np)
        b_min_e = np.zeros(p, dtype=dtype_np)
        ik_min_e = np.ones(p, dtype=dtype_np)
        a_min_h = np.zeros(p, dtype=dtype_np)
        b_min_h = np.zeros(p, dtype=dtype_np)
        ik_min_h = np.ones(p, dtype=dtype_np)

    # Max 側 (インデックス n-p .. n-1)
    idx_max = np.arange(p, dtype=np.float64)
    d_max_e = idx_max / float(p)
    d_max_h = (idx_max + 0.5) / float(p)
    if max_is_pml:
        a_max_e, b_max_e, ik_max_e = _compute_abk(d_max_e)
        a_max_h, b_max_h, ik_max_h = _compute_abk(d_max_h)
    else:
        a_max_e = np.zeros(p, dtype=dtype_np)
        b_max_e = np.zeros(p, dtype=dtype_np)
        ik_max_e = np.ones(p, dtype=dtype_np)
        a_max_h = np.zeros(p, dtype=dtype_np)
        b_max_h = np.zeros(p, dtype=dtype_np)
        ik_max_h = np.ones(p, dtype=dtype_np)

    ik_e = np.ones(n, dtype=dtype_np)
    ik_h = np.ones(n, dtype=dtype_np)
    if min_is_pml:
        ik_e[:p] = ik_min_e
        ik_h[:p] = ik_min_h
    if max_is_pml:
        ik_e[-p:] = ik_max_e
        ik_h[-p:] = ik_max_h

    return (
        a_min_e,
        b_min_e,
        a_max_e,
        b_max_e,
        ik_e,
        a_min_h,
        b_min_h,
        a_max_h,
        b_max_h,
        ik_h,
    )


# ---------------------------------------------------------------------------
# スライスベース 3D Yee 有限差分ヘルパー (jnp.roll 不使用・厳密 PEC/PMC 対応)
# ---------------------------------------------------------------------------
def _diff_bwd_x(arr: jnp.ndarray, px: int) -> jnp.ndarray:
    """x軸 (axis=2) 後方差分。px=+1(PMC): 2*arr[0], px=-1(PEC): 0, px=0: arr[0]。"""
    if px == 1:
        b0 = 2.0 * arr[:, :, 0:1]
    elif px == -1:
        b0 = jnp.zeros_like(arr[:, :, 0:1])
    else:
        b0 = arr[:, :, 0:1]
    return jnp.concatenate([b0, arr[:, :, 1:] - arr[:, :, :-1]], axis=2)


def _diff_fwd_x(arr: jnp.ndarray) -> jnp.ndarray:
    """x軸 (axis=2) 前進差分 (外端は Dirichlet 0)。"""
    return jnp.concatenate([arr[:, :, 1:] - arr[:, :, :-1], -arr[:, :, -1:]], axis=2)


def _diff_bwd_z(arr: jnp.ndarray, pz: int) -> jnp.ndarray:
    """z軸 (axis=0) 後方差分。pz=+1(PMC): 2*arr[0], pz=-1(PEC): 0, pz=0: arr[0]。"""
    if pz == 1:
        b0 = 2.0 * arr[0:1, :, :]
    elif pz == -1:
        b0 = jnp.zeros_like(arr[0:1, :, :])
    else:
        b0 = arr[0:1, :, :]
    return jnp.concatenate([b0, arr[1:, :, :] - arr[:-1, :, :]], axis=0)


def _diff_fwd_z(arr: jnp.ndarray) -> jnp.ndarray:
    """z軸 (axis=0) 前進差分 (外端は Dirichlet 0)。"""
    return jnp.concatenate([arr[1:, :, :] - arr[:-1, :, :], -arr[-1:, :, :]], axis=0)


def _diff_bwd_y(arr: jnp.ndarray, py: int, sharded: bool = False, n_shards: int = 1) -> jnp.ndarray:
    """y軸 (axis=1) 後方差分。sharded=True 時は lax.ppermute で左隣シャード末尾を受け取る。"""
    rest = arr[:, 1:, :] - arr[:, :-1, :]
    if py == 1:
        b0_edge = 2.0 * arr[:, 0:1, :]
    elif py == -1:
        b0_edge = jnp.zeros_like(arr[:, 0:1, :])
    else:
        b0_edge = arr[:, 0:1, :]

    if not sharded:
        return jnp.concatenate([b0_edge, rest], axis=1)

    perm = [(i, (i + 1) % n_shards) for i in range(n_shards)]
    recv_left = jax.lax.ppermute(arr[:, -1:, :], axis_name="y", perm=perm)
    is_first = jax.lax.axis_index("y") == 0
    b0 = jnp.where(is_first, b0_edge, arr[:, 0:1, :] - recv_left)
    return jnp.concatenate([b0, rest], axis=1)


def _diff_fwd_y(arr: jnp.ndarray, sharded: bool = False, n_shards: int = 1) -> jnp.ndarray:
    """y軸 (axis=1) 前進差分。sharded=True 時は lax.ppermute で右隣シャード先頭を受け取る。"""
    interior = arr[:, 1:, :] - arr[:, :-1, :]
    if not sharded:
        return jnp.concatenate([interior, -arr[:, -1:, :]], axis=1)

    perm = [(i, (i - 1) % n_shards) for i in range(n_shards)]
    recv_right = jax.lax.ppermute(arr[:, 0:1, :], axis_name="y", perm=perm)
    is_last = jax.lax.axis_index("y") == (n_shards - 1)
    blast = jnp.where(is_last, -arr[:, -1:, :], recv_right - arr[:, -1:, :])
    return jnp.concatenate([interior, blast], axis=1)


def _apply_cpml_axis(
    dF: jnp.ndarray,
    psi_min: jnp.ndarray,
    psi_max: jnp.ndarray,
    a_min: jnp.ndarray,
    b_min: jnp.ndarray,
    a_max: jnp.ndarray,
    b_max: jnp.ndarray,
    ik_full: jnp.ndarray,
    axis: int,
    min_active: bool,
    max_active: bool,
    pml_cells: int,
    sharded_y: bool = False,
    n_shards: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    境界スラブ CPML を適用する。
    重要: 必ず未スケーリングの生差分スラブで psi を更新してから ik_full を乗算し、最後に psi を加算する。
    """
    p = pml_cells
    if axis == 2:
        if min_active:
            s_min = dF[:, :, :p]
            psi_min = b_min[None, None, :] * psi_min + a_min[None, None, :] * s_min
        if max_active:
            s_max = dF[:, :, -p:]
            psi_max = b_max[None, None, :] * psi_max + a_max[None, None, :] * s_max
        dF = dF * ik_full[None, None, :]
        if min_active:
            dF = dF.at[:, :, :p].add(psi_min)
        if max_active:
            dF = dF.at[:, :, -p:].add(psi_max)
        return dF, psi_min, psi_max

    elif axis == 0:
        if min_active:
            s_min = dF[:p, :, :]
            psi_min = b_min[:, None, None] * psi_min + a_min[:, None, None] * s_min
        if max_active:
            s_max = dF[-p:, :, :]
            psi_max = b_max[:, None, None] * psi_max + a_max[:, None, None] * s_max
        dF = dF * ik_full[:, None, None]
        if min_active:
            dF = dF.at[:p, :, :].add(psi_min)
        if max_active:
            dF = dF.at[-p:, :, :].add(psi_max)
        return dF, psi_min, psi_max

    else:  # axis == 1 (y)
        if sharded_y:
            is_first = jax.lax.axis_index("y") == 0
            is_last = jax.lax.axis_index("y") == (n_shards - 1)
            if min_active:
                s_min = dF[:, :p, :]
                psi_min_new = b_min[None, :, None] * psi_min + a_min[None, :, None] * s_min
                psi_min = jnp.where(is_first, psi_min_new, psi_min)
            if max_active:
                s_max = dF[:, -p:, :]
                psi_max_new = b_max[None, :, None] * psi_max + a_max[None, :, None] * s_max
                psi_max = jnp.where(is_last, psi_max_new, psi_max)
            dF = dF * ik_full[None, :, None]
            if min_active:
                dF = dF.at[:, :p, :].add(jnp.where(is_first, psi_min, jnp.zeros_like(psi_min)))
            if max_active:
                dF = dF.at[:, -p:, :].add(jnp.where(is_last, psi_max, jnp.zeros_like(psi_max)))
            return dF, psi_min, psi_max
        else:
            if min_active:
                s_min = dF[:, :p, :]
                psi_min = b_min[None, :, None] * psi_min + a_min[None, :, None] * s_min
            if max_active:
                s_max = dF[:, -p:, :]
                psi_max = b_max[None, :, None] * psi_max + a_max[None, :, None] * s_max
            dF = dF * ik_full[None, :, None]
            if min_active:
                dF = dF.at[:, :p, :].add(psi_min)
            if max_active:
                dF = dF.at[:, -p:, :].add(psi_max)
            return dF, psi_min, psi_max


def _run_fdtdx_fallback_simulation(
    eps_np_xyz: np.ndarray,
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    res: int,
    fcen: float,
    df: float,
    t_max: float,
    src_positions: List[Tuple[float, float, float]],
    src_comps: List[str],
    src_phases: np.ndarray,
    mon_positions: List[Tuple[float, float, float]],
    mon_comps: List[str],
    pml_cells: int,
    px: int,
    py: int,
    pz: int,
    cutoff: float,
    stride: int,
    chunk_steps: int,
    decay_threshold: float,
    dft_freqs: Optional[List[float]],
    dtype_jnp: Any,
    progress_callback: Optional[Callable[[int, int, float, float], None]],
) -> Optional[TPUSimulationResult]:
    """`pml_mode='fdtdx'` 指定時かつ FDTDX が利用可能で <=8M セルの場合のオプション実行経路。"""
    try:
        import fdtdx
        from fdtdx.fdtd.update import update_E as fdtdx_update_E
        from fdtdx.fdtd.update import update_H as fdtdx_update_H
    except Exception:
        return None

    _, nx, ny, nz = eps_np_xyz.shape
    if nx * ny * nz > 8_000_000:
        return None

    xs, ys, zs = coords
    dx = 1.0 / float(res)
    courant_num = 0.99 / math.sqrt(3.0)
    dt_meep = courant_num * dx
    t_src_end = 2.0 * cutoff / max(df, 1e-9)
    t0_pulse = 0.5 * t_src_end
    pulse_w = 1.0 / max(df, 1e-9)
    total_steps_max = max(100, int(math.ceil(t_max / dt_meep)))
    stride = max(1, int(stride))
    chunk_steps = max(stride * 10, (int(chunk_steps) // stride) * stride)
    samples_per_chunk = chunk_steps // stride

    inv_eps_dev = jnp.asarray(1.0 / np.maximum(eps_np_xyz, 1.0), dtype=dtype_jnp)

    try:
        spacing_m = float(dx) * 1e-6
        dt_si = 0.99 * spacing_m / (math.sqrt(3.0) * 299792458.0)
        sim_cfg = fdtdx.SimulationConfig(
            time=dt_si * 10.0, resolution=spacing_m, dtype=dtype_jnp, courant_factor=0.99
        )
        volume = fdtdx.SimulationVolume(
            partial_grid_shape=(nx, ny, nz), material=fdtdx.Material(permittivity=1.0)
        )
        overrides = build_fdtdx_boundary_override(px=px, py=py, pz=pz)
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=pml_cells, boundary_type="pml", override_types=overrides if overrides else None
        )
        bound_objs, constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
        objects, arrays, _, sim_cfg, _ = fdtdx.place_objects(
            object_list=[volume] + list(bound_objs.values()),
            config=sim_cfg,
            constraints=constraints,
            key=jax.random.PRNGKey(0),
        )
        arrays = arrays.aset("inv_permittivities", inv_eps_dev)
    except Exception:
        return None

    e_src_list, h_src_list = [], []
    for p_idx, (sx_p, sy_p, sz_p) in enumerate(src_positions):
        ix = _coord_to_index(xs, abs(sx_p) if px != 0 else sx_p)
        iy = _coord_to_index(ys, abs(sy_p) if py != 0 else sy_p)
        iz = _coord_to_index(zs, abs(sz_p) if pz != 0 else sz_p)
        for c_idx, cname in enumerate(src_comps):
            eh_type, comp_i = COMP_TO_IDX[cname]
            phi = float(src_phases[p_idx % len(src_phases), c_idx % 6])
            (e_src_list if eh_type == "E" else h_src_list).append((comp_i, ix, iy, iz, phi))

    mon_ix = jnp.asarray([_coord_to_index(xs, abs(p[0]) if px != 0 else p[0]) for p in mon_positions], dtype=jnp.int32)
    mon_iy = jnp.asarray([_coord_to_index(ys, abs(p[1]) if py != 0 else p[1]) for p in mon_positions], dtype=jnp.int32)
    mon_iz = jnp.asarray([_coord_to_index(zs, abs(p[2]) if pz != 0 else p[2]) for p in mon_positions], dtype=jnp.int32)
    mon_eh_flags = [0 if COMP_TO_IDX[c][0] == "E" else 1 for c in mon_comps]
    mon_comp_indices = [COMP_TO_IDX[c][1] for c in mon_comps]

    has_dft = bool(dft_freqs) and len(dft_freqs) > 0
    dft_freqs_list = list(dft_freqs) if has_dft else []
    n_freqs = len(dft_freqs_list)
    dft_freqs_j = jnp.asarray(dft_freqs_list, dtype=jnp.float32) if has_dft else jnp.zeros((1,), dtype=jnp.float32)

    iz_z0 = _coord_to_index(zs, 0.0)
    iz_zoff = _coord_to_index(zs, 0.06 if pz != 0 else -0.06)
    iy_y0 = _coord_to_index(ys, 0.0)
    ix_x0 = _coord_to_index(xs, 0.0)

    @jax.jit(donate_argnums=(1, 2))
    def _run_chunk_fdtdx(start_step: jnp.ndarray, carry_state, dft_acc):
        def _outer(c_tup, sample_idx):
            st, dft_buf = c_tup
            base_step = start_step + sample_idx * stride

            def _inner(arrs, sub_k):
                step_i = base_step + sub_k
                t_cur = step_i.astype(jnp.float32) * float(dt_meep)
                env = jnp.exp(-0.5 * ((t_cur - float(t0_pulse)) / float(pulse_w)) ** 2) * jnp.where(
                    t_cur <= float(t_src_end), 1.0, 0.0
                )
                arrs = fdtdx_update_E(step_i, arrs, objects, sim_cfg, simulate_boundaries=True)
                E_cur = arrs.E
                for comp_i, ix, iy, iz, phi in e_src_list:
                    val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                    E_cur = E_cur.at[comp_i, ix, iy, iz].add(val)
                arrs = arrs.aset("E", apply_e_mirror_bc(E_cur, px=px, py=py, pz=pz))

                arrs = fdtdx_update_H(step_i, arrs, objects, sim_cfg, simulate_boundaries=True)
                H_cur = arrs.H
                for comp_i, ix, iy, iz, phi in h_src_list:
                    val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                    H_cur = H_cur.at[comp_i, ix, iy, iz].add(val)
                arrs = arrs.aset("H", apply_h_mirror_bc(H_cur, px=px, py=py, pz=pz))
                return arrs, None

            st_after, _ = jax.lax.scan(_inner, st, jnp.arange(stride, dtype=jnp.int32))
            E_now, H_now = st_after.E, st_after.H
            cols = [
                (E_now if f == 0 else H_now)[ci, mon_ix, mon_iy, mon_iz].astype(jnp.float32)
                for f, ci in zip(mon_eh_flags, mon_comp_indices)
            ]
            mon_sample = jnp.stack(cols, axis=-1)

            if has_dft:
                t_now = (base_step + stride - 1).astype(jnp.float32) * float(dt_meep)
                active_mask = jnp.where(t_now > float(t_src_end), 1.0, 0.0)
                frac = dft_freqs_j * t_now - jnp.floor(dft_freqs_j * t_now)
                angle = -2.0 * math.pi * frac
                phases = (jnp.cos(angle) + 1j * jnp.sin(angle)).astype(jnp.complex64) * active_mask
                eh_z0 = jnp.concatenate([E_now[:, :, :, iz_z0], H_now[:, :, :, iz_z0]], axis=0).astype(jnp.complex64)
                eh_zoff = jnp.concatenate([E_now[:, :, :, iz_zoff], H_now[:, :, :, iz_zoff]], axis=0).astype(jnp.complex64)
                eh_y0 = jnp.concatenate([E_now[:, :, iy_y0, :], H_now[:, :, iy_y0, :]], axis=0).astype(jnp.complex64)
                eh_x0 = jnp.concatenate([E_now[:, ix_x0, :, :], H_now[:, ix_x0, :, :]], axis=0).astype(jnp.complex64)
                b_z0, b_zoff, b_y0, b_x0 = dft_buf
                dft_buf = (
                    b_z0 + phases[:, None, None, None] * eh_z0[None, ...],
                    b_zoff + phases[:, None, None, None] * eh_zoff[None, ...],
                    b_y0 + phases[:, None, None, None] * eh_y0[None, ...],
                    b_x0 + phases[:, None, None, None] * eh_x0[None, ...],
                )
            return (st_after, dft_buf), mon_sample

        (final_st, final_dft), chunk_mon_traces = jax.lax.scan(
            _outer, (carry_state, dft_acc), jnp.arange(samples_per_chunk, dtype=jnp.int32)
        )
        return final_st, final_dft, chunk_mon_traces

    carry = arrays
    dft_acc = (
        (
            jnp.zeros((n_freqs, 6, nx, ny), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, nx, ny), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, nx, nz), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, ny, nz), dtype=jnp.complex64),
        )
        if has_dft
        else ()
    )

    all_traces: List[np.ndarray] = []
    peak_energy = 0.0
    final_energy = 0.0
    stopped_early = False
    current_step = 0
    t_start_wall = time.perf_counter()
    jit_compile_sec = 0.0

    while current_step < total_steps_max:
        t0 = time.perf_counter()
        carry, dft_acc, traces_dev = _run_chunk_fdtdx(jnp.asarray(current_step, dtype=jnp.int32), carry, dft_acc)
        traces_np = np.asarray(traces_dev, dtype=np.float32)
        del traces_dev
        t1 = time.perf_counter()
        if current_step == 0:
            jit_compile_sec = t1 - t0
        all_traces.append(traces_np)
        current_step += chunk_steps
        t_cur_meep = current_step * dt_meep
        tail_len = max(1, samples_per_chunk // 4)
        chunk_energy = float(np.mean(traces_np[-tail_len:] ** 2))
        if not np.isfinite(chunk_energy):
            raise RuntimeError("FDTD simulation produced NaN/Inf field values.")
        peak_energy = max(peak_energy, chunk_energy)
        final_energy = chunk_energy
        if progress_callback is not None:
            progress_callback(current_step, total_steps_max, t_cur_meep, final_energy)
        if t_cur_meep > t_src_end + 80.0 / max(fcen, 0.1) and peak_energy > 0.0:
            if final_energy < peak_energy * decay_threshold:
                stopped_early = True
                break

    wall_time_sec = time.perf_counter() - t_start_wall
    eff_sim_sec = max(wall_time_sec - jit_compile_sec, 1e-6)
    mcells_per_sec = (float(nx) * float(ny) * float(nz) * float(current_step) / 1e6) / eff_sim_sec
    full_traces = np.concatenate(all_traces, axis=0)
    dt_rec = float(stride) * dt_meep
    time_array = (np.arange(len(full_traces), dtype=np.float64) + 1.0) * dt_rec

    dft_results: Dict[int, Dict[str, Any]] = {}
    if has_dft:
        b_z0_np, b_zoff_np, b_y0_np, b_x0_np = [np.asarray(x) for x in dft_acc]
        eps_np = eps_np_xyz[0]
        eps_z0, eps_zoff = eps_np[:, :, iz_z0], eps_np[:, :, iz_zoff]
        eps_y0, eps_x0 = eps_np[:, iy_y0, :], eps_np[ix_x0, :, :]
        eps_y0_full, zs_full = unfold_2d_slice_z(eps_y0, zs, pz=pz, comp_name=None)
        eps_x0_full, _ = unfold_2d_slice_z(eps_x0, zs, pz=pz, comp_name=None)

        for f_idx, freq_val in enumerate(dft_freqs_list):
            def _dens(eh_s, eps_s, is_z):
                d_d = 0.5 * eps_s * np.sum(np.abs(eh_s[0:3]) ** 2, axis=0)
                h_d = 0.5 * np.sum(np.abs(eh_s[3:6]) ** 2, axis=0)
                if is_z and pz != 0:
                    d_d, _ = unfold_2d_slice_z(d_d, zs, pz=pz, comp_name=None)
                    h_d, _ = unfold_2d_slice_z(h_d, zs, pz=pz, comp_name=None)
                return d_d, h_d

            d_z0, h_z0 = _dens(b_z0_np[f_idx], eps_z0, False)
            d_zoff, h_zoff = _dens(b_zoff_np[f_idx], eps_zoff, False)
            d_y0, h_y0 = _dens(b_y0_np[f_idx], eps_y0, True)
            d_x0, h_x0 = _dens(b_x0_np[f_idx], eps_x0, True)
            dft_results[f_idx] = {
                "freq": freq_val, "xs": xs, "ys": ys, "zs_full": zs_full,
                "eps_xy_z0": eps_z0, "eps_xy_zoff": eps_zoff,
                "eps_xz_y0": eps_y0_full, "eps_yz_x0": eps_x0_full,
                "Dden_xy_z0": d_z0, "Hden_xy_z0": h_z0,
                "Dden_xy_zoff": d_zoff, "Hden_xy_zoff": h_zoff,
                "Dden_xz_y0": d_y0, "Hden_xz_y0": h_y0,
                "Dden_yz_x0": d_x0, "Hden_yz_x0": h_x0,
                "fields_xy_z0": b_z0_np[f_idx],
            }

    del carry, dft_acc, inv_eps_dev
    return TPUSimulationResult(
        time_array=time_array, monitor_traces=full_traces,
        monitor_positions=mon_positions, monitor_components=mon_comps,
        dt_meep=dt_meep, dt_rec=dt_rec, t_src_end=t_src_end,
        total_steps_run=current_step, stopped_early_by_decay=stopped_early,
        peak_energy=peak_energy, final_energy=final_energy,
        wall_time_sec=wall_time_sec, jit_compile_sec=jit_compile_sec,
        mcells_per_sec=mcells_per_sec, grid_shape=(nx, ny, nz), coords=coords,
        pz=pz, dft_slices=dft_results, dft_freqs=dft_freqs_list,
        pml_mode="fdtdx", n_shards=1, device_kind=detect_tpu_hardware()["device_kind"],
    )


def _get_or_compile_chunk_fn(
    cache_key: Tuple[Any, ...],
    nx: int,
    ny: int,
    nz: int,
    pml_cells: int,
    px: int,
    py: int,
    pz: int,
    stride: int,
    samples_per_chunk: int,
    scan_unroll: int,
    pml_mode: str,
    use_sharding: bool,
    n_shards: int,
    has_dft: bool,
    n_freqs: int,
    aniso_eps: bool,
    e_src_meta: Tuple[Tuple[int, int], ...],
    h_src_meta: Tuple[Tuple[int, int], ...],
    mon_eh_flags: Tuple[int, ...],
    mon_comp_indices: Tuple[int, ...],
    iz_z0: int,
    iz_zoff: int,
    iy_y0: int,
    ix_x0: int,
    dtype_jnp: Any,
    mesh: Optional[Mesh],
) -> Callable:
    """
    キャッシュキーに一致するコンパイル済み `_run_chunk` を返す。
    同一形状・構成のステップ (例: STEP 2.5 -> STEP 3) では再コンパイルゼロで実行できる。
    """
    if cache_key in _JIT_CHUNK_CACHE:
        fn = _JIT_CHUNK_CACHE.pop(cache_key)
        _JIT_CHUNK_CACHE[cache_key] = fn
        return fn

    ny_loc = ny // n_shards if use_sharding else ny
    y_owner = iy_y0 // ny_loc if use_sharding else 0
    iy_y0_loc = iy_y0 % ny_loc if use_sharding else iy_y0

    def _chunk_body(
        start_step: jnp.ndarray,
        carry_state,
        dft_acc,
        inv_eps_dev: jnp.ndarray,
        pml_tuple,
        src_packed,
        mon_packed,
        dft_freqs_j: jnp.ndarray,
        pulse_params: jnp.ndarray,
    ):
        dt_meep_j = pulse_params[0]
        fcen_j = pulse_params[1]
        t0_pulse_j = pulse_params[2]
        pulse_w_j = pulse_params[3]
        t_src_end_j = pulse_params[4]
        courant_j = pulse_params[5].astype(dtype_jnp)

        e_src_coords, e_src_phases = src_packed[0], src_packed[1]
        h_src_coords, h_src_phases = src_packed[2], src_packed[3]
        mon_ix, mon_iy, mon_iz = mon_packed

        if use_sharding:
            shard_idx = jax.lax.axis_index("y")
            is_y_first = shard_idx == 0
            # sharded 時、psi_y_* と b_y0 は先頭にサイズ1のシャード軸を持つためアンパック/パックする
            E_cur, H_cur, cpml_raw = carry_state
            if pml_mode == "cpml":
                px_s, py_s_sh, pz_s = cpml_raw
                py_s = tuple(arr[0] for arr in py_s_sh)
                cpml_local = (px_s, py_s, pz_s)
            else:
                cpml_local = cpml_raw
            if has_dft:
                b_z0_i, b_zoff_i, b_y0_sh, b_x0_i = dft_acc
                dft_local = (b_z0_i, b_zoff_i, b_y0_sh[0], b_x0_i)
            else:
                dft_local = dft_acc
            carry_local = (E_cur, H_cur, cpml_local)
        else:
            shard_idx = 0
            is_y_first = True
            carry_local = carry_state
            dft_local = dft_acc

        # 誘電率の逆数: 等方かつ pz != 0 のとき Ez (z=(k+0.5)*dz) 用にオンザフライで z 補間
        if aniso_eps:
            inv_ex, inv_ey, inv_ez = inv_eps_dev[0], inv_eps_dev[1], inv_eps_dev[2]
        else:
            inv_e0 = inv_eps_dev[0]
            inv_ex = inv_e0
            inv_ey = inv_e0
            if pz != 0:
                inv_ez = jnp.concatenate([0.5 * (inv_e0[:-1] + inv_e0[1:]), inv_e0[-1:]], axis=0)
            else:
                inv_ez = inv_e0

        def _single_step(step_i: jnp.ndarray, st_in):
            (Ex, Ey, Ez), (Hx, Hy, Hz), cpml_st = st_in
            t_cur = step_i.astype(jnp.float32) * dt_meep_j
            env = jnp.exp(-0.5 * ((t_cur - t0_pulse_j) / pulse_w_j) ** 2) * jnp.where(
                t_cur <= t_src_end_j, 1.0, 0.0
            )
            phase_base = 2.0 * math.pi * fcen_j * t_cur

            # 1) curl_H (後方差分) -> E 更新
            dHz_dy = _diff_bwd_y(Hz, py=py, sharded=use_sharding, n_shards=n_shards)
            dHy_dz = _diff_bwd_z(Hy, pz=pz)
            dHx_dz = _diff_bwd_z(Hx, pz=pz)
            dHz_dx = _diff_bwd_x(Hz, px=px)
            dHy_dx = _diff_bwd_x(Hy, px=px)
            dHx_dy = _diff_bwd_y(Hx, py=py, sharded=use_sharding, n_shards=n_shards)

            if pml_mode == "cpml":
                (ax_min_e, bx_min_e, ax_max_e, bx_max_e, ikx_e, ax_min_h, bx_min_h, ax_max_h, bx_max_h, ikx_h) = pml_tuple[0]
                (ay_min_e, by_min_e, ay_max_e, by_max_e, iky_e, ay_min_h, by_min_h, ay_max_h, by_max_h, iky_h) = pml_tuple[1]
                (az_min_e, bz_min_e, az_max_e, bz_max_e, ikz_e, az_min_h, bz_min_h, az_max_h, bz_max_h, ikz_h) = pml_tuple[2]

                (psi_x_me, psi_x_xe, psi_x_mh, psi_x_xh) = cpml_st[0]
                (psi_y_me, psi_y_xe, psi_y_mh, psi_y_xh) = cpml_st[1]
                (psi_z_me, psi_z_xe, psi_z_mh, psi_z_xh) = cpml_st[2]

                dHz_dx, psi_x_me0, psi_x_xe0 = _apply_cpml_axis(
                    dHz_dx, psi_x_me[0], psi_x_xe[0], ax_min_e, bx_min_e, ax_max_e, bx_max_e, ikx_e,
                    axis=2, min_active=(px == 0), max_active=True, pml_cells=pml_cells,
                )
                dHy_dx, psi_x_me1, psi_x_xe1 = _apply_cpml_axis(
                    dHy_dx, psi_x_me[1], psi_x_xe[1], ax_min_e, bx_min_e, ax_max_e, bx_max_e, ikx_e,
                    axis=2, min_active=(px == 0), max_active=True, pml_cells=pml_cells,
                )
                psi_x_me = jnp.stack([psi_x_me0, psi_x_me1], axis=0) if px == 0 else psi_x_me
                psi_x_xe = jnp.stack([psi_x_xe0, psi_x_xe1], axis=0)

                dHz_dy, psi_y_me0, psi_y_xe0 = _apply_cpml_axis(
                    dHz_dy, psi_y_me[0], psi_y_xe[0], ay_min_e, by_min_e, ay_max_e, by_max_e, iky_e,
                    axis=1, min_active=(py == 0), max_active=True, pml_cells=pml_cells,
                    sharded_y=use_sharding, n_shards=n_shards,
                )
                dHx_dy, psi_y_me1, psi_y_xe1 = _apply_cpml_axis(
                    dHx_dy, psi_y_me[1], psi_y_xe[1], ay_min_e, by_min_e, ay_max_e, by_max_e, iky_e,
                    axis=1, min_active=(py == 0), max_active=True, pml_cells=pml_cells,
                    sharded_y=use_sharding, n_shards=n_shards,
                )
                psi_y_me = jnp.stack([psi_y_me0, psi_y_me1], axis=0) if py == 0 else psi_y_me
                psi_y_xe = jnp.stack([psi_y_xe0, psi_y_xe1], axis=0)

                dHy_dz, psi_z_me0, psi_z_xe0 = _apply_cpml_axis(
                    dHy_dz, psi_z_me[0], psi_z_xe[0], az_min_e, bz_min_e, az_max_e, bz_max_e, ikz_e,
                    axis=0, min_active=(pz == 0), max_active=True, pml_cells=pml_cells,
                )
                dHx_dz, psi_z_me1, psi_z_xe1 = _apply_cpml_axis(
                    dHx_dz, psi_z_me[1], psi_z_xe[1], az_min_e, bz_min_e, az_max_e, bz_max_e, ikz_e,
                    axis=0, min_active=(pz == 0), max_active=True, pml_cells=pml_cells,
                )
                psi_z_me = jnp.stack([psi_z_me0, psi_z_me1], axis=0) if pz == 0 else psi_z_me
                psi_z_xe = jnp.stack([psi_z_xe0, psi_z_xe1], axis=0)

                Ex = Ex + courant_j * (dHz_dy - dHy_dz) * inv_ex
                Ey = Ey + courant_j * (dHx_dz - dHz_dx) * inv_ey
                Ez = Ez + courant_j * (dHy_dx - dHx_dy) * inv_ez
            else:
                damp_z, damp_y, damp_x = pml_tuple
                damp_xyz = damp_z * damp_y * damp_x
                Ex = (Ex + courant_j * (dHz_dy - dHy_dz) * inv_ex) * damp_xyz
                Ey = (Ey + courant_j * (dHx_dz - dHz_dx) * inv_ey) * damp_xyz
                Ez = (Ez + courant_j * (dHy_dx - dHx_dy) * inv_ez) * damp_xyz

            # E 点双極子励振源の注入
            E_list = [Ex, Ey, Ez]
            for s_idx, (comp_i, owner_sh) in enumerate(e_src_meta):
                iz_s, iy_loc_s, ix_s = e_src_coords[s_idx, 0], e_src_coords[s_idx, 1], e_src_coords[s_idx, 2]
                val = (env * jnp.sin(phase_base + e_src_phases[s_idx])).astype(dtype_jnp)
                if use_sharding:
                    val = jnp.where(shard_idx == owner_sh, val, jnp.asarray(0.0, dtype=dtype_jnp))
                E_list[comp_i] = E_list[comp_i].at[iz_s, iy_loc_s, ix_s].add(val)

            Ex, Ey, Ez = apply_e_mirror_bc_zyx(
                E_list[0], E_list[1], E_list[2], px=px, py=py, pz=pz, is_y_first_shard=is_y_first
            )

            # 2) curl_E (前進差分) -> H 更新
            dEz_dy = _diff_fwd_y(Ez, sharded=use_sharding, n_shards=n_shards)
            dEy_dz = _diff_fwd_z(Ey)
            dEx_dz = _diff_fwd_z(Ex)
            dEz_dx = _diff_fwd_x(Ez)
            dEy_dx = _diff_fwd_x(Ey)
            dEx_dy = _diff_fwd_y(Ex, sharded=use_sharding, n_shards=n_shards)

            if pml_mode == "cpml":
                dEz_dx, psi_x_mh0, psi_x_xh0 = _apply_cpml_axis(
                    dEz_dx, psi_x_mh[0], psi_x_xh[0], ax_min_h, bx_min_h, ax_max_h, bx_max_h, ikx_h,
                    axis=2, min_active=(px == 0), max_active=True, pml_cells=pml_cells,
                )
                dEy_dx, psi_x_mh1, psi_x_xh1 = _apply_cpml_axis(
                    dEy_dx, psi_x_mh[1], psi_x_xh[1], ax_min_h, bx_min_h, ax_max_h, bx_max_h, ikx_h,
                    axis=2, min_active=(px == 0), max_active=True, pml_cells=pml_cells,
                )
                psi_x_mh = jnp.stack([psi_x_mh0, psi_x_mh1], axis=0) if px == 0 else psi_x_mh
                psi_x_xh = jnp.stack([psi_x_xh0, psi_x_xh1], axis=0)

                dEz_dy, psi_y_mh0, psi_y_xh0 = _apply_cpml_axis(
                    dEz_dy, psi_y_mh[0], psi_y_xh[0], ay_min_h, by_min_h, ay_max_h, by_max_h, iky_h,
                    axis=1, min_active=(py == 0), max_active=True, pml_cells=pml_cells,
                    sharded_y=use_sharding, n_shards=n_shards,
                )
                dEx_dy, psi_y_mh1, psi_y_xh1 = _apply_cpml_axis(
                    dEx_dy, psi_y_mh[1], psi_y_xh[1], ay_min_h, by_min_h, ay_max_h, by_max_h, iky_h,
                    axis=1, min_active=(py == 0), max_active=True, pml_cells=pml_cells,
                    sharded_y=use_sharding, n_shards=n_shards,
                )
                psi_y_mh = jnp.stack([psi_y_mh0, psi_y_mh1], axis=0) if py == 0 else psi_y_mh
                psi_y_xh = jnp.stack([psi_y_xh0, psi_y_xh1], axis=0)

                dEy_dz, psi_z_mh0, psi_z_xh0 = _apply_cpml_axis(
                    dEy_dz, psi_z_mh[0], psi_z_xh[0], az_min_h, bz_min_h, az_max_h, bz_max_h, ikz_h,
                    axis=0, min_active=(pz == 0), max_active=True, pml_cells=pml_cells,
                )
                dEx_dz, psi_z_mh1, psi_z_xh1 = _apply_cpml_axis(
                    dEx_dz, psi_z_mh[1], psi_z_xh[1], az_min_h, bz_min_h, az_max_h, bz_max_h, ikz_h,
                    axis=0, min_active=(pz == 0), max_active=True, pml_cells=pml_cells,
                )
                psi_z_mh = jnp.stack([psi_z_mh0, psi_z_mh1], axis=0) if pz == 0 else psi_z_mh
                psi_z_xh = jnp.stack([psi_z_xh0, psi_z_xh1], axis=0)

                Hx = Hx - courant_j * (dEz_dy - dEy_dz)
                Hy = Hy - courant_j * (dEx_dz - dEz_dx)
                Hz = Hz - courant_j * (dEy_dx - dEx_dy)
                cpml_out = (
                    (psi_x_me, psi_x_xe, psi_x_mh, psi_x_xh),
                    (psi_y_me, psi_y_xe, psi_y_mh, psi_y_xh),
                    (psi_z_me, psi_z_xe, psi_z_mh, psi_z_xh),
                )
            else:
                damp_z, damp_y, damp_x = pml_tuple
                damp_xyz = damp_z * damp_y * damp_x
                Hx = (Hx - courant_j * (dEz_dy - dEy_dz)) * damp_xyz
                Hy = (Hy - courant_j * (dEx_dz - dEz_dx)) * damp_xyz
                Hz = (Hz - courant_j * (dEy_dx - dEx_dy)) * damp_xyz
                cpml_out = cpml_st

            # H 点双極子励振源の注入
            H_list = [Hx, Hy, Hz]
            for s_idx, (comp_i, owner_sh) in enumerate(h_src_meta):
                iz_s, iy_loc_s, ix_s = h_src_coords[s_idx, 0], h_src_coords[s_idx, 1], h_src_coords[s_idx, 2]
                val = (env * jnp.sin(phase_base + h_src_phases[s_idx])).astype(dtype_jnp)
                if use_sharding:
                    val = jnp.where(shard_idx == owner_sh, val, jnp.asarray(0.0, dtype=dtype_jnp))
                H_list[comp_i] = H_list[comp_i].at[iz_s, iy_loc_s, ix_s].add(val)

            Hx, Hy, Hz = apply_h_mirror_bc_zyx(
                H_list[0], H_list[1], H_list[2], px=px, py=py, pz=pz, is_y_first_shard=is_y_first
            )

            return ((Ex, Ey, Ez), (Hx, Hy, Hz), cpml_out)

        def _sample_monitors(E_tup, H_tup) -> jnp.ndarray:
            if use_sharding:
                owner = mon_iy // ny_loc
                iy_loc = mon_iy % ny_loc
                is_owner = (shard_idx == owner)
                cols = []
                for eh_flag, c_idx in zip(mon_eh_flags, mon_comp_indices):
                    arr = E_tup[c_idx] if eh_flag == 0 else H_tup[c_idx]
                    vals = arr[mon_iz, iy_loc, mon_ix].astype(jnp.float32)
                    cols.append(jnp.where(is_owner, vals, 0.0))
                local_mat = jnp.stack(cols, axis=-1)
                return jax.lax.psum(local_mat, axis_name="y")
            else:
                cols = []
                for eh_flag, c_idx in zip(mon_eh_flags, mon_comp_indices):
                    arr = E_tup[c_idx] if eh_flag == 0 else H_tup[c_idx]
                    cols.append(arr[mon_iz, mon_iy, mon_ix].astype(jnp.float32))
                return jnp.stack(cols, axis=-1)

        def _outer_step(c_tuple, sample_idx):
            st, dft_buf = c_tuple
            base_step = start_step + sample_idx * stride

            def _inner_step(st_in, sub_k):
                return _single_step(base_step + sub_k, st_in), None

            st_after, _ = jax.lax.scan(
                _inner_step,
                st,
                jnp.arange(stride, dtype=jnp.int32),
                unroll=scan_unroll,
            )
            E_now, H_now, _ = st_after
            mon_sample = _sample_monitors(E_now, H_now)

            if has_dft:
                t_now = (base_step + stride - 1).astype(jnp.float32) * dt_meep_j
                active_mask = jnp.where(t_now > t_src_end_j, 1.0, 0.0)
                frac = dft_freqs_j * t_now - jnp.floor(dft_freqs_j * t_now)
                angle = -2.0 * math.pi * frac
                phases = (jnp.cos(angle) + 1j * jnp.sin(angle)).astype(jnp.complex64) * active_mask

                eh_z0 = jnp.stack(
                    [
                        E_now[0][iz_z0], E_now[1][iz_z0], E_now[2][iz_z0],
                        H_now[0][iz_z0], H_now[1][iz_z0], H_now[2][iz_z0],
                    ],
                    axis=0,
                ).astype(jnp.complex64)
                eh_zoff = jnp.stack(
                    [
                        E_now[0][iz_zoff], E_now[1][iz_zoff], E_now[2][iz_zoff],
                        H_now[0][iz_zoff], H_now[1][iz_zoff], H_now[2][iz_zoff],
                    ],
                    axis=0,
                ).astype(jnp.complex64)
                eh_x0 = jnp.stack(
                    [
                        E_now[0][:, :, ix_x0], E_now[1][:, :, ix_x0], E_now[2][:, :, ix_x0],
                        H_now[0][:, :, ix_x0], H_now[1][:, :, ix_x0], H_now[2][:, :, ix_x0],
                    ],
                    axis=0,
                ).astype(jnp.complex64)

                b_z0, b_zoff, b_y0, b_x0 = dft_buf
                b_z0 = b_z0 + phases[:, None, None, None] * eh_z0[None, ...]
                b_zoff = b_zoff + phases[:, None, None, None] * eh_zoff[None, ...]
                b_x0 = b_x0 + phases[:, None, None, None] * eh_x0[None, ...]

                eh_y0 = jnp.stack(
                    [
                        E_now[0][:, iy_y0_loc, :], E_now[1][:, iy_y0_loc, :], E_now[2][:, iy_y0_loc, :],
                        H_now[0][:, iy_y0_loc, :], H_now[1][:, iy_y0_loc, :], H_now[2][:, iy_y0_loc, :],
                    ],
                    axis=0,
                ).astype(jnp.complex64)
                if use_sharding:
                    y0_mask = jnp.where(shard_idx == y_owner, 1.0, 0.0).astype(jnp.complex64)
                    b_y0 = b_y0 + (phases[:, None, None, None] * eh_y0[None, ...]) * y0_mask
                else:
                    b_y0 = b_y0 + phases[:, None, None, None] * eh_y0[None, ...]

                dft_buf = (b_z0, b_zoff, b_y0, b_x0)

            return (st_after, dft_buf), mon_sample

        (final_st_loc, final_dft_loc), chunk_mon_traces = jax.lax.scan(
            _outer_step,
            (carry_local, dft_local),
            jnp.arange(samples_per_chunk, dtype=jnp.int32),
        )

        if use_sharding:
            E_f, H_f, cpml_f = final_st_loc
            if pml_mode == "cpml":
                px_f, py_f, pz_f = cpml_f
                py_f_sh = tuple(arr[None, ...] for arr in py_f)
                cpml_out = (px_f, py_f_sh, pz_f)
            else:
                cpml_out = cpml_f
            if has_dft:
                bz0_f, bzoff_f, by0_f, bx0_f = final_dft_loc
                dft_out = (bz0_f, bzoff_f, by0_f[None, ...], bx0_f)
            else:
                dft_out = final_dft_loc
            return (E_f, H_f, cpml_out), dft_out, chunk_mon_traces
        else:
            return final_st_loc, final_dft_loc, chunk_mon_traces

    if use_sharding:
        if _jax_shard_map is None or mesh is None:
            raise RuntimeError("shard_map is unavailable in this JAX installation.")

        field_spec = P(None, "y", None)
        eh_spec = (field_spec, field_spec, field_spec)

        if pml_mode == "cpml":
            px_active_spec = P(None, None, "y", None)
            px_dummy_spec = P()
            px_spec = (
                px_active_spec if px == 0 else px_dummy_spec,
                px_active_spec,
                px_active_spec if px == 0 else px_dummy_spec,
                px_active_spec,
            )
            py_active_spec = P("y", None, None, None, None)
            py_dummy_spec = P()
            py_spec = (
                py_active_spec if py == 0 else py_dummy_spec,
                py_active_spec,
                py_active_spec if py == 0 else py_dummy_spec,
                py_active_spec,
            )
            pz_active_spec = P(None, None, "y", None)
            pz_dummy_spec = P()
            pz_spec = (
                pz_active_spec if pz == 0 else pz_dummy_spec,
                pz_active_spec,
                pz_active_spec if pz == 0 else pz_dummy_spec,
                pz_active_spec,
            )
            cpml_spec = (px_spec, py_spec, pz_spec)

            pml_x_spec = (P(),) * 10
            pml_y_spec = (P(), P(), P(), P(), P("y"), P(), P(), P(), P(), P("y"))
            pml_z_spec = (P(),) * 10
            pml_tuple_spec = (pml_x_spec, pml_y_spec, pml_z_spec)
        else:
            cpml_spec = ()
            pml_tuple_spec = (P(), P(None, "y", None), P())

        carry_spec = (eh_spec, eh_spec, cpml_spec)
        if has_dft:
            dft_spec = (
                P(None, None, "y", None),
                P(None, None, "y", None),
                P("y", None, None, None, None),
                P(None, None, None, "y"),
            )
        else:
            dft_spec = ()

        in_specs = (
            P(),  # start_step
            carry_spec,
            dft_spec,
            P(None, None, "y", None),  # inv_eps_dev
            pml_tuple_spec,
            (P(), P(), P(), P()),  # src_packed
            (P(), P(), P()),  # mon_packed
            P(),  # dft_freqs_j
            P(),  # pulse_params
        )
        out_specs = (carry_spec, dft_spec, P())

        sharded_fn = _jax_shard_map(
            _chunk_body,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

        @jax.jit(donate_argnums=(1, 2))
        def _run_chunk(start_step, carry_state, dft_acc, inv_eps_dev, pml_tuple, src_packed, mon_packed, dft_freqs_j, pulse_params):
            return sharded_fn(start_step, carry_state, dft_acc, inv_eps_dev, pml_tuple, src_packed, mon_packed, dft_freqs_j, pulse_params)
    else:
        @jax.jit(donate_argnums=(1, 2))
        def _run_chunk(start_step, carry_state, dft_acc, inv_eps_dev, pml_tuple, src_packed, mon_packed, dft_freqs_j, pulse_params):
            return _chunk_body(start_step, carry_state, dft_acc, inv_eps_dev, pml_tuple, src_packed, mon_packed, dft_freqs_j, pulse_params)

    if len(_JIT_CHUNK_CACHE) >= _MAX_JIT_CACHE_ENTRIES:
        _JIT_CHUNK_CACHE.popitem(last=False)
    _JIT_CHUNK_CACHE[cache_key] = _run_chunk
    return _run_chunk


def run_tpu_simulation(
    eps_grid: np.ndarray | jnp.ndarray,
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    res: int,
    fcen: float,
    df: float,
    t_max: float,
    src_positions: List[Tuple[float, float, float]],
    src_comps: List[str],
    src_phases: np.ndarray,
    mon_positions: List[Tuple[float, float, float]],
    mon_comps: List[str],
    pml_cells: int = 12,
    px: int = 0,
    py: int = 0,
    pz: int = 0,
    cutoff: float = 5.0,
    stride: int = 2,
    chunk_steps: int = 2000,
    decay_threshold: float = 1e-5,
    dft_freqs: Optional[List[float]] = None,
    dtype_str: str = "float32",
    enable_x64: bool = False,
    allow_tpu_x64: bool = False,
    pml_mode: str = "cpml",
    sharding: str = "auto",
    scan_unroll: int = 1,
    keep_jit_cache: bool = True,
    xla_cache_dir: Optional[str] = "/tmp/spv_tpu_xla_cache",
    layout: str = "xyz",
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> TPUSimulationResult:
    """
    TPU 最適化 3D FDTD カーネルを用いて時間領域シミュレーションを実行する。
    """
    if xla_cache_dir:
        configure_tpu_xla_cache(xla_cache_dir)

    if not keep_jit_cache:
        clear_jit_cache()

    hw = detect_tpu_hardware()
    is_tpu = bool(hw["is_tpu"])

    if dtype_str == "bfloat16":
        warnings.warn(
            "bfloat16 has 7-bit mantissa; resonant phase / Q-factor accuracy may degrade compared to float32."
        )
        dtype_jnp = jnp.bfloat16
    elif enable_x64:
        if is_tpu and not allow_tpu_x64:
            warnings.warn(
                "TPU v5e/v6e lacks native hardware float64; keeping float32 unless allow_tpu_x64=True."
            )
            dtype_jnp = jnp.float32
        else:
            jax.config.update("jax_enable_x64", True)
            dtype_jnp = jnp.float64
    else:
        dtype_jnp = jnp.float32

    # 入力 eps_grid の正規化 (layout='zyx' ならゼロコピー、'xyz' なら (C, Nz, Ny, Nx) へ変換)
    eps_host = np.asarray(eps_grid, dtype=np.float32)
    if isinstance(eps_grid, jax.Array):
        try:
            eps_grid.delete()
        except Exception:
            pass
    del eps_grid
    gc.collect()

    layout_l = str(layout).lower()
    if layout_l == "xyz":
        # (C, Nx, Ny, Nz) -> (C, Nz, Ny, Nx)
        eps_zyx_np = np.ascontiguousarray(np.transpose(eps_host, (0, 3, 2, 1)))
        eps_xyz_np = eps_host
    elif layout_l == "zyx":
        eps_zyx_np = eps_host
        eps_xyz_np = None
    else:
        raise ValueError(f"Unsupported layout: {layout!r}. Expected 'zyx' or 'xyz'.")

    n_eps_comp, nz, ny, nx = eps_zyx_np.shape
    pml_mode_l = str(pml_mode).lower()

    if pml_mode_l == "fdtdx":
        if eps_xyz_np is None:
            eps_xyz_np = np.ascontiguousarray(np.transpose(eps_zyx_np, (0, 3, 2, 1)))
        fdtdx_res = _run_fdtdx_fallback_simulation(
            eps_np_xyz=eps_xyz_np,
            coords=coords, res=res, fcen=fcen, df=df, t_max=t_max,
            src_positions=src_positions, src_comps=src_comps, src_phases=src_phases,
            mon_positions=mon_positions, mon_comps=mon_comps,
            pml_cells=pml_cells, px=px, py=py, pz=pz, cutoff=cutoff,
            stride=stride, chunk_steps=chunk_steps, decay_threshold=decay_threshold,
            dft_freqs=dft_freqs, dtype_jnp=dtype_jnp, progress_callback=progress_callback,
        )
        if fdtdx_res is not None:
            return fdtdx_res
        warnings.warn("pml_mode='fdtdx' requested but fdtdx unavailable or >8M cells; falling back to 'cpml'.")
        pml_mode_l = "cpml"

    xs, ys, zs = coords
    dx = 1.0 / float(res)
    courant_num = 0.99 / math.sqrt(3.0)
    dt_meep = courant_num * dx

    t_src_end = 2.0 * cutoff / max(df, 1e-9)
    t0_pulse = 0.5 * t_src_end
    pulse_w = 1.0 / max(df, 1e-9)

    total_steps_max = max(100, int(math.ceil(t_max / dt_meep)))
    stride = max(1, int(stride))
    chunk_steps = max(stride * 10, (int(chunk_steps) // stride) * stride)
    samples_per_chunk = chunk_steps // stride
    pml_cells = max(2, int(pml_cells))

    # マルチチップ TPU y 軸 sharding の可否判定
    devs = jax.devices()
    num_devices = len(devs)
    sharding_l = str(sharding).lower()
    use_sharding = (
        sharding_l != "off"
        and num_devices > 1
        and _jax_shard_map is not None
        and (ny % num_devices == 0)
        and ((ny // num_devices) >= max(16, pml_cells))
    )
    n_shards = num_devices if use_sharding else 1
    ny_loc = ny // n_shards
    mesh = Mesh(np.array(devs), ("y",)) if use_sharding else None

    # 励振源のインデックスと座標を準備
    e_src_meta_list: List[Tuple[int, int]] = []
    e_src_coords_list: List[Tuple[int, int, int]] = []
    e_src_phases_list: List[float] = []
    h_src_meta_list: List[Tuple[int, int]] = []
    h_src_coords_list: List[Tuple[int, int, int]] = []
    h_src_phases_list: List[float] = []

    for p_idx, (sx_p, sy_p, sz_p) in enumerate(src_positions):
        ix = _coord_to_index(xs, abs(sx_p) if px != 0 else sx_p)
        iy_g = _coord_to_index(ys, abs(sy_p) if py != 0 else sy_p)
        iz = _coord_to_index(zs, abs(sz_p) if pz != 0 else sz_p)
        owner_sh = iy_g // ny_loc if use_sharding else 0
        iy_l = iy_g % ny_loc if use_sharding else iy_g
        for c_idx, cname in enumerate(src_comps):
            eh_type, comp_i = COMP_TO_IDX[cname]
            phi = float(src_phases[p_idx % len(src_phases), c_idx % 6])
            if eh_type == "E":
                e_src_meta_list.append((comp_i, owner_sh))
                e_src_coords_list.append((iz, iy_l, ix))
                e_src_phases_list.append(phi)
            else:
                h_src_meta_list.append((comp_i, owner_sh))
                h_src_coords_list.append((iz, iy_l, ix))
                h_src_phases_list.append(phi)

    e_src_coords_j = jnp.asarray(e_src_coords_list if e_src_coords_list else [(0, 0, 0)], dtype=jnp.int32)
    e_src_phases_j = jnp.asarray(e_src_phases_list if e_src_phases_list else [0.0], dtype=jnp.float32)
    h_src_coords_j = jnp.asarray(h_src_coords_list if h_src_coords_list else [(0, 0, 0)], dtype=jnp.int32)
    h_src_phases_j = jnp.asarray(h_src_phases_list if h_src_phases_list else [0.0], dtype=jnp.float32)
    src_packed = (e_src_coords_j, e_src_phases_j, h_src_coords_j, h_src_phases_j)

    # モニタ点のインデックス
    mon_ix = jnp.asarray([_coord_to_index(xs, abs(p[0]) if px != 0 else p[0]) for p in mon_positions], dtype=jnp.int32)
    mon_iy = jnp.asarray([_coord_to_index(ys, abs(p[1]) if py != 0 else p[1]) for p in mon_positions], dtype=jnp.int32)
    mon_iz = jnp.asarray([_coord_to_index(zs, abs(p[2]) if pz != 0 else p[2]) for p in mon_positions], dtype=jnp.int32)
    mon_packed = (mon_ix, mon_iy, mon_iz)
    mon_eh_flags = tuple(0 if COMP_TO_IDX[c][0] == "E" else 1 for c in mon_comps)
    mon_comp_indices = tuple(COMP_TO_IDX[c][1] for c in mon_comps)

    # Running DFT 断面準備
    has_dft = bool(dft_freqs) and len(dft_freqs) > 0
    dft_freqs_list = list(dft_freqs) if has_dft else []
    n_freqs = len(dft_freqs_list)
    dft_freqs_j = jnp.asarray(dft_freqs_list, dtype=jnp.float32) if has_dft else jnp.zeros((1,), dtype=jnp.float32)

    iz_z0 = _coord_to_index(zs, 0.0)
    iz_zoff = _coord_to_index(zs, 0.06 if pz != 0 else -0.06)
    iy_y0 = _coord_to_index(ys, 0.0)
    ix_x0 = _coord_to_index(xs, 0.0)
    y_owner = iy_y0 // ny_loc if use_sharding else 0

    aniso_eps = n_eps_comp == 3
    inv_eps_np = (1.0 / np.maximum(eps_zyx_np, 1.0)).astype(np.float32)

    # デバイスへ配置
    if use_sharding:
        sh_field = NamedSharding(mesh, P(None, "y", None))
        sh_eps = NamedSharding(mesh, P(None, None, "y", None))
        inv_eps_dev = jax.device_put(jnp.asarray(inv_eps_np, dtype=dtype_jnp), sh_eps)
    else:
        inv_eps_dev = jnp.asarray(inv_eps_np, dtype=dtype_jnp)

    # PML 係数 & 状態構築
    if pml_mode_l == "cpml":
        cx = _build_cpml_1d_coeffs(nx, pml_cells, dx, dt_meep, min_is_pml=(px == 0), max_is_pml=True)
        cy = _build_cpml_1d_coeffs(ny, pml_cells, dx, dt_meep, min_is_pml=(py == 0), max_is_pml=True)
        cz = _build_cpml_1d_coeffs(nz, pml_cells, dx, dt_meep, min_is_pml=(pz == 0), max_is_pml=True)

        cx_j = tuple(jnp.asarray(a, dtype=dtype_jnp) for a in cx)
        cz_j = tuple(jnp.asarray(a, dtype=dtype_jnp) for a in cz)
        if use_sharding:
            sh_1d_y = NamedSharding(mesh, P("y"))
            cy_j = tuple(
                jax.device_put(jnp.asarray(a, dtype=dtype_jnp), sh_1d_y)
                if i in (4, 9)
                else jnp.asarray(a, dtype=dtype_jnp)
                for i, a in enumerate(cy)
            )
        else:
            cy_j = tuple(jnp.asarray(a, dtype=dtype_jnp) for a in cy)
        pml_tuple = (cx_j, cy_j, cz_j)

        def _alloc_slab(shape, spec_p):
            arr = jnp.zeros(shape, dtype=dtype_jnp)
            return jax.device_put(arr, NamedSharding(mesh, spec_p)) if use_sharding else arr

        psi_x = (
            _alloc_slab((2, nz, ny, pml_cells) if px == 0 else (1, 1, 1, 1), P(None, None, "y", None) if px == 0 else P()),
            _alloc_slab((2, nz, ny, pml_cells), P(None, None, "y", None)),
            _alloc_slab((2, nz, ny, pml_cells) if px == 0 else (1, 1, 1, 1), P(None, None, "y", None) if px == 0 else P()),
            _alloc_slab((2, nz, ny, pml_cells), P(None, None, "y", None)),
        )
        if use_sharding:
            psi_y = (
                _alloc_slab((n_shards, 2, nz, pml_cells, nx) if py == 0 else (1, 1, 1, 1, 1), P("y", None, None, None, None) if py == 0 else P()),
                _alloc_slab((n_shards, 2, nz, pml_cells, nx), P("y", None, None, None, None)),
                _alloc_slab((n_shards, 2, nz, pml_cells, nx) if py == 0 else (1, 1, 1, 1, 1), P("y", None, None, None, None) if py == 0 else P()),
                _alloc_slab((n_shards, 2, nz, pml_cells, nx), P("y", None, None, None, None)),
            )
        else:
            psi_y = (
                _alloc_slab((2, nz, pml_cells, nx) if py == 0 else (1, 1, 1, 1), P()),
                _alloc_slab((2, nz, pml_cells, nx), P()),
                _alloc_slab((2, nz, pml_cells, nx) if py == 0 else (1, 1, 1, 1), P()),
                _alloc_slab((2, nz, pml_cells, nx), P()),
            )
        psi_z = (
            _alloc_slab((2, pml_cells, ny, nx) if pz == 0 else (1, 1, 1, 1), P(None, None, "y", None) if pz == 0 else P()),
            _alloc_slab((2, pml_cells, ny, nx), P(None, None, "y", None)),
            _alloc_slab((2, pml_cells, ny, nx) if pz == 0 else (1, 1, 1, 1), P(None, None, "y", None) if pz == 0 else P()),
            _alloc_slab((2, pml_cells, ny, nx), P(None, None, "y", None)),
        )
        cpml_init = (psi_x, psi_y, psi_z)
    else:
        damp_z = jnp.asarray(_build_sponge_1d_profile(nz, pml_cells, min_is_pml=(pz == 0), max_is_pml=True), dtype=dtype_jnp)[:, None, None]
        damp_y_np = _build_sponge_1d_profile(ny, pml_cells, min_is_pml=(py == 0), max_is_pml=True)[None, :, None]
        damp_y = jax.device_put(jnp.asarray(damp_y_np, dtype=dtype_jnp), NamedSharding(mesh, P(None, "y", None))) if use_sharding else jnp.asarray(damp_y_np, dtype=dtype_jnp)
        damp_x = jnp.asarray(_build_sponge_1d_profile(nx, pml_cells, min_is_pml=(px == 0), max_is_pml=True), dtype=dtype_jnp)[None, None, :]
        pml_tuple = (damp_z, damp_y, damp_x)
        cpml_init = ()

    def _zeros_field():
        z_arr = jnp.zeros((nz, ny, nx), dtype=dtype_jnp)
        return jax.device_put(z_arr, sh_field) if use_sharding else z_arr

    E_init = (_zeros_field(), _zeros_field(), _zeros_field())
    H_init = (_zeros_field(), _zeros_field(), _zeros_field())
    carry = (E_init, H_init, cpml_init)

    if has_dft:
        if use_sharding:
            dft_acc = (
                jax.device_put(jnp.zeros((n_freqs, 6, ny, nx), dtype=jnp.complex64), NamedSharding(mesh, P(None, None, "y", None))),
                jax.device_put(jnp.zeros((n_freqs, 6, ny, nx), dtype=jnp.complex64), NamedSharding(mesh, P(None, None, "y", None))),
                jax.device_put(jnp.zeros((n_shards, n_freqs, 6, nz, nx), dtype=jnp.complex64), NamedSharding(mesh, P("y", None, None, None, None))),
                jax.device_put(jnp.zeros((n_freqs, 6, nz, ny), dtype=jnp.complex64), NamedSharding(mesh, P(None, None, None, "y"))),
            )
        else:
            dft_acc = (
                jnp.zeros((n_freqs, 6, ny, nx), dtype=jnp.complex64),
                jnp.zeros((n_freqs, 6, ny, nx), dtype=jnp.complex64),
                jnp.zeros((n_freqs, 6, nz, nx), dtype=jnp.complex64),
                jnp.zeros((n_freqs, 6, nz, ny), dtype=jnp.complex64),
            )
    else:
        dft_acc = ()

    pulse_params = jnp.asarray([dt_meep, fcen, t0_pulse, pulse_w, t_src_end, courant_num], dtype=jnp.float32)

    e_src_meta = tuple(e_src_meta_list)
    h_src_meta = tuple(h_src_meta_list)
    cache_key = (
        nx, ny, nz, pml_cells, px, py, pz, dtype_str, pml_mode_l,
        use_sharding, n_shards, stride, chunk_steps, int(scan_unroll),
        has_dft, n_freqs, aniso_eps,
        e_src_meta, h_src_meta, mon_eh_flags, mon_comp_indices,
        iz_z0, iz_zoff, iy_y0, ix_x0,
    )
    run_chunk_fn = _get_or_compile_chunk_fn(
        cache_key=cache_key,
        nx=nx, ny=ny, nz=nz, pml_cells=pml_cells,
        px=px, py=py, pz=pz, stride=stride,
        samples_per_chunk=samples_per_chunk, scan_unroll=max(1, int(scan_unroll)),
        pml_mode=pml_mode_l, use_sharding=use_sharding, n_shards=n_shards,
        has_dft=has_dft, n_freqs=n_freqs, aniso_eps=aniso_eps,
        e_src_meta=e_src_meta, h_src_meta=h_src_meta,
        mon_eh_flags=mon_eh_flags, mon_comp_indices=mon_comp_indices,
        iz_z0=iz_z0, iz_zoff=iz_zoff, iy_y0=iy_y0, ix_x0=ix_x0,
        dtype_jnp=dtype_jnp, mesh=mesh,
    )

    all_traces: List[np.ndarray] = []
    peak_energy = 0.0
    final_energy = 0.0
    stopped_early = False
    current_step = 0

    t_start_wall = time.perf_counter()
    jit_compile_sec = 0.0

    try:
        while current_step < total_steps_max:
            t_c0 = time.perf_counter()
            carry, dft_acc, traces_dev = run_chunk_fn(
                jnp.asarray(current_step, dtype=jnp.int32),
                carry,
                dft_acc,
                inv_eps_dev,
                pml_tuple,
                src_packed,
                mon_packed,
                dft_freqs_j,
                pulse_params,
            )
            traces_np = np.asarray(traces_dev, dtype=np.float32)
            del traces_dev
            t_c1 = time.perf_counter()

            if current_step == 0:
                jit_compile_sec = t_c1 - t_c0

            all_traces.append(traces_np)
            current_step += chunk_steps
            t_cur_meep = current_step * dt_meep

            tail_len = max(1, samples_per_chunk // 4)
            chunk_energy = float(np.mean(traces_np[-tail_len:] ** 2))
            if not np.isfinite(chunk_energy) or not np.all(np.isfinite(traces_np)):
                raise RuntimeError("FDTD simulation produced NaN/Inf field values.")
            if chunk_energy > peak_energy:
                peak_energy = chunk_energy
            final_energy = chunk_energy

            if progress_callback is not None:
                progress_callback(current_step, total_steps_max, t_cur_meep, final_energy)

            min_ringdown_time = t_src_end + 80.0 / max(fcen, 0.1)
            if t_cur_meep > min_ringdown_time and peak_energy > 0.0:
                if final_energy < peak_energy * decay_threshold:
                    stopped_early = True
                    break

        wall_time_sec = time.perf_counter() - t_start_wall
        eff_sim_sec = max(wall_time_sec - jit_compile_sec, 1e-6)
        mcells_per_sec = (float(nx) * float(ny) * float(nz) * float(current_step) / 1e6) / eff_sim_sec

        full_traces = np.concatenate(all_traces, axis=0)
        dt_rec = float(stride) * dt_meep
        time_array = (np.arange(len(full_traces), dtype=np.float64) + 1.0) * dt_rec

        dft_results: Dict[int, Dict[str, Any]] = {}
        if has_dft:
            # (Ny, Nx) -> (Nx, Ny), (Nz, Nx) -> (Nx, Nz), (Nz, Ny) -> (Ny, Nz) に統一
            b_z0_np = np.transpose(np.asarray(dft_acc[0]), (0, 1, 3, 2))
            b_zoff_np = np.transpose(np.asarray(dft_acc[1]), (0, 1, 3, 2))
            b_y0_raw = np.asarray(dft_acc[2])[y_owner] if use_sharding else np.asarray(dft_acc[2])
            b_y0_np = np.transpose(b_y0_raw, (0, 1, 3, 2))
            b_x0_np = np.transpose(np.asarray(dft_acc[3]), (0, 1, 3, 2))

            eps_z0 = eps_zyx_np[0, iz_z0, :, :].T
            eps_zoff = eps_zyx_np[0, iz_zoff, :, :].T
            eps_y0 = eps_zyx_np[0, :, iy_y0, :].T
            eps_x0 = eps_zyx_np[0, :, :, ix_x0].T

            eps_y0_full, zs_full = unfold_2d_slice_z(eps_y0, zs, pz=pz, comp_name=None)
            eps_x0_full, _ = unfold_2d_slice_z(eps_x0, zs, pz=pz, comp_name=None)

            for f_idx, freq_val in enumerate(dft_freqs_list):
                def _compute_densities(eh_slice: np.ndarray, eps_slice: np.ndarray, is_z_slice: bool):
                    e_sq = np.sum(np.abs(eh_slice[0:3]) ** 2, axis=0)
                    h_sq = np.sum(np.abs(eh_slice[3:6]) ** 2, axis=0)
                    d_den = 0.5 * eps_slice * e_sq
                    h_den = 0.5 * h_sq
                    if is_z_slice and pz != 0:
                        d_den, _ = unfold_2d_slice_z(d_den, zs, pz=pz, comp_name=None)
                        h_den, _ = unfold_2d_slice_z(h_den, zs, pz=pz, comp_name=None)
                    return d_den, h_den

                d_z0, h_z0 = _compute_densities(b_z0_np[f_idx], eps_z0, is_z_slice=False)
                d_zoff, h_zoff = _compute_densities(b_zoff_np[f_idx], eps_zoff, is_z_slice=False)
                d_y0, h_y0 = _compute_densities(b_y0_np[f_idx], eps_y0, is_z_slice=True)
                d_x0, h_x0 = _compute_densities(b_x0_np[f_idx], eps_x0, is_z_slice=True)

                dft_results[f_idx] = {
                    "freq": freq_val,
                    "xs": xs,
                    "ys": ys,
                    "zs_full": zs_full,
                    "eps_xy_z0": eps_z0,
                    "eps_xy_zoff": eps_zoff,
                    "eps_xz_y0": eps_y0_full,
                    "eps_yz_x0": eps_x0_full,
                    "Dden_xy_z0": d_z0,
                    "Hden_xy_z0": h_z0,
                    "Dden_xy_zoff": d_zoff,
                    "Hden_xy_zoff": h_zoff,
                    "Dden_xz_y0": d_y0,
                    "Hden_xz_y0": h_y0,
                    "Dden_yz_x0": d_x0,
                    "Hden_yz_x0": h_x0,
                    "fields_xy_z0": b_z0_np[f_idx],
                }
    finally:
        del carry, dft_acc, inv_eps_dev, pml_tuple, src_packed, mon_packed, dft_freqs_j, pulse_params
        if not keep_jit_cache:
            clear_jit_cache()
        else:
            gc.collect()

    return TPUSimulationResult(
        time_array=time_array,
        monitor_traces=full_traces,
        monitor_positions=mon_positions,
        monitor_components=mon_comps,
        dt_meep=dt_meep,
        dt_rec=dt_rec,
        t_src_end=t_src_end,
        total_steps_run=current_step,
        stopped_early_by_decay=stopped_early,
        peak_energy=peak_energy,
        final_energy=final_energy,
        wall_time_sec=wall_time_sec,
        jit_compile_sec=jit_compile_sec,
        mcells_per_sec=mcells_per_sec,
        grid_shape=(nx, ny, nz),
        coords=coords,
        pz=pz,
        dft_slices=dft_results,
        dft_freqs=dft_freqs_list,
        pml_mode=pml_mode_l,
        n_shards=n_shards,
        device_kind=hw["device_kind"],
    )


run_fdtdx_simulation = run_tpu_simulation
