"""
FDTDX (JAX/GPU) simulation runner (Spec B1 §4-6).
Features:
- Uses `fdtdx` (SimulationConfig, BoundaryConfig, SimulationVolume, PerfectlyMatchedLayer,
  update_E, update_H) on JAX GPU with automatic memory-saving native JAX 3D Yee+PML kernel.
- Uses `@jax.jit(donate_argnums=...)` buffer donation and separable 1D PML profiles to minimize GPU VRAM.
- Chunked `jax.lax.scan` execution to avoid Python loop overhead while checking field energy decay.
- Multi-point, multi-component simultaneous Gaussian pulse dipole sources with fixed random phases.
- Strided point time-series buffer inside `lax.scan` transferred per chunk for Harminv CLI.
- Running 2D cross-section DFT (xy at z=0, xy at z=-0.06, xz at y=0, yz at x=0) for STEP 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from .symmetry import (
    COMP_TO_IDX,
    apply_e_mirror_bc,
    apply_h_mirror_bc,
    build_fdtdx_boundary_override,
    unfold_2d_slice_z,
)

try:
    import fdtdx
    from fdtdx.fdtd.update import update_E as fdtdx_update_E
    from fdtdx.fdtd.update import update_H as fdtdx_update_H

    HAS_FDTDX = True
except Exception:
    fdtdx = None
    HAS_FDTDX = False


# JIT キャッシュクリア関数 (Spec B1 §5)
_JIT_CHUNK_CACHE: Dict[Tuple[Any, ...], Callable] = {}


def clear_jit_cache() -> None:
    """JIT キャッシュおよび JAX/XLA のコンパイル済み実行グラフキャッシュを解放する。"""
    _JIT_CHUNK_CACHE.clear()
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def check_gpu_environment(
    nx: int,
    ny: int,
    nz: int,
    dtype: str = "float32",
    require_gpu: bool = False,
) -> Dict[str, Any]:
    """
    JAX の利用可能デバイス・FDTDX の有無・推定 VRAM 使用量を事前チェックする (Spec B1 §2 / §5)。
    """
    devices = jax.devices()
    backend_platform = devices[0].platform if devices else "cpu"
    device_kind = getattr(devices[0], "device_kind", str(devices[0])) if devices else "unknown"

    if require_gpu and backend_platform != "gpu":
        raise RuntimeError(
            f"GPU が検出されませんでした (現在のバックエンド: {backend_platform})。"
            "Google Colab の「ランタイム > ランタイムのタイプを変更」から GPU (T4 / L4 / A100) を選択してください。"
        )

    bytes_per_elem = 2 if dtype == "bfloat16" else (8 if dtype == "float64" else 4)
    est_gb = (float(nx) * float(ny) * float(nz) * bytes_per_elem * 16.0) / (1024.0**3)

    return {
        "platform": backend_platform,
        "device_kind": device_kind,
        "num_devices": len(devices),
        "has_fdtdx": HAS_FDTDX,
        "fdtdx_version": getattr(fdtdx, "__version__", "fallback_jax") if HAS_FDTDX else "fallback_jax",
        "grid_shape": (nx, ny, nz),
        "total_cells": nx * ny * nz,
        "estimated_vram_gb": est_gb,
    }


@dataclass
class FDTDXSimulationResult:
    """1回の FDTDX 実行結果を保持するデータクラス。"""

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


def _coord_to_index(coord_1d: np.ndarray, val: float) -> int:
    """1D 座標配列から最も近いセルインデックスを返す。"""
    return int(np.clip(np.argmin(np.abs(coord_1d - val)), 0, len(coord_1d) - 1))


def _build_pure_jax_pml_profile(n: int, pml_cells: int, min_is_pml: bool = True, max_is_pml: bool = True) -> np.ndarray:
    """
    1D 減衰係数プロファイル exp(-σ(i)) を生成する (3次多項式グレーディング)。
    3D 配列へ展開せず 1D ベクトルのまま保持することで VRAM 消費をゼロに抑える。
    """
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


def _setup_fdtdx_state(
    inv_eps: jnp.ndarray,
    dx_um: float,
    pml_cells: int,
    px: int,
    py: int,
    pz: int,
    dtype_jnp: Any,
    max_cells_for_fdtdx_container: int = 8_000_000,
):
    """
    グリッドサイズが十分小さい場合は FDTDX の ArrayContainer を使い、
    中〜大規模 3D グリッド (> 8M セル) では PML 補助場の重複確保による VRAM 枯渇を防ぐため
    省メモリのネイティブ JAX Yee カーネルを選択する。
    """
    _, nx, ny, nz = inv_eps.shape
    total_cells = int(nx) * int(ny) * int(nz)

    if HAS_FDTDX and total_cells <= max_cells_for_fdtdx_container:
        try:
            spacing_m = float(dx_um) * 1e-6
            courant_factor = 0.99
            dt_si = courant_factor * spacing_m / (math.sqrt(3.0) * 299792458.0)
            sim_cfg = fdtdx.SimulationConfig(
                time=dt_si * 10.0,
                resolution=spacing_m,
                dtype=dtype_jnp,
                courant_factor=courant_factor,
            )
            volume = fdtdx.SimulationVolume(
                partial_grid_shape=(nx, ny, nz),
                material=fdtdx.Material(permittivity=1.0),
            )
            overrides = build_fdtdx_boundary_override(px=px, py=py, pz=pz)
            bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
                thickness=pml_cells,
                boundary_type="pml",
                override_types=overrides if overrides else None,
            )
            bound_objs, constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
            obj_list = [volume] + list(bound_objs.values())
            key = jax.random.PRNGKey(0)
            objects, arrays, _, sim_cfg, _ = fdtdx.place_objects(
                object_list=obj_list,
                config=sim_cfg,
                constraints=constraints,
                key=key,
            )
            arrays = arrays.aset("inv_permittivities", inv_eps)
            return True, sim_cfg, objects, arrays
        except Exception as exc:
            print(f"[spv_gpu] Info: Using memory-optimized JAX Yee+PML kernel ({exc})")

    return False, None, None, inv_eps


def run_fdtdx_simulation(
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
    chunk_steps: int = 1000,
    decay_threshold: float = 1e-5,
    dft_freqs: Optional[List[float]] = None,
    dtype_str: str = "float32",
    enable_x64: bool = False,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> FDTDXSimulationResult:
    """
    FDTDX (JAX/GPU) を用いて時間領域 FDTD シミュレーションを実行する。
    """
    # 前回のコンパイルキャッシュ・未回収デバイスバッファを解放して BFC OOM を防止
    clear_jit_cache()

    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        dtype_jnp = jnp.float64
    elif dtype_str == "bfloat16":
        dtype_jnp = jnp.bfloat16
    else:
        dtype_jnp = jnp.float32

    # eps_grid はホスト (CPU) NumPy 配列として保持し、GPU には inv_eps のみを配置する
    eps_np_full = np.asarray(eps_grid, dtype=np.float32)
    if isinstance(eps_grid, jax.Array):
        try:
            eps_grid.delete()
        except Exception:
            pass
    del eps_grid
    gc.collect()

    xs, ys, zs = coords
    _, nx, ny, nz = eps_np_full.shape
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

    # 励振源のグリッドインデックスと成分・位相を準備
    e_src_list = []
    h_src_list = []
    for p_idx, (sx_p, sy_p, sz_p) in enumerate(src_positions):
        ix = _coord_to_index(xs, abs(sx_p) if px != 0 else sx_p)
        iy = _coord_to_index(ys, abs(sy_p) if py != 0 else sy_p)
        iz = _coord_to_index(zs, abs(sz_p) if pz != 0 else sz_p)
        for c_idx, cname in enumerate(src_comps):
            eh_type, comp_i = COMP_TO_IDX[cname]
            phi = float(src_phases[p_idx % len(src_phases), c_idx % 6])
            if eh_type == "E":
                e_src_list.append((comp_i, ix, iy, iz, phi))
            else:
                h_src_list.append((comp_i, ix, iy, iz, phi))

    # モニタ点のグリッドインデックスを準備
    mon_ix = jnp.asarray([_coord_to_index(xs, abs(p[0]) if px != 0 else p[0]) for p in mon_positions], dtype=jnp.int32)
    mon_iy = jnp.asarray([_coord_to_index(ys, abs(p[1]) if py != 0 else p[1]) for p in mon_positions], dtype=jnp.int32)
    mon_iz = jnp.asarray([_coord_to_index(zs, abs(p[2]) if pz != 0 else p[2]) for p in mon_positions], dtype=jnp.int32)

    mon_eh_flags = [0 if COMP_TO_IDX[c][0] == "E" else 1 for c in mon_comps]
    mon_comp_indices = [COMP_TO_IDX[c][1] for c in mon_comps]

    # Running DFT 断面インデックス準備 (STEP 3)
    has_dft = bool(dft_freqs) and len(dft_freqs) > 0
    dft_freqs_list = list(dft_freqs) if has_dft else []
    n_freqs = len(dft_freqs_list)
    dft_freqs_j = jnp.asarray(dft_freqs_list, dtype=jnp.float32) if has_dft else jnp.zeros((1,), dtype=jnp.float32)

    iz_z0 = _coord_to_index(zs, 0.0)
    iz_zoff = _coord_to_index(zs, 0.06 if pz != 0 else -0.06)
    iy_y0 = _coord_to_index(ys, 0.0)
    ix_x0 = _coord_to_index(xs, 0.0)

    # 1 / eps のみを GPU に転送
    inv_eps_dev = jnp.asarray(1.0 / np.maximum(eps_np_full, 1.0), dtype=dtype_jnp)

    use_fdtdx_kernel, sim_cfg, objects, arrays_or_inveps = _setup_fdtdx_state(
        inv_eps=inv_eps_dev,
        dx_um=dx,
        pml_cells=pml_cells,
        px=px,
        py=py,
        pz=pz,
        dtype_jnp=dtype_jnp,
    )

    # 1D 分離型 PML 減衰係数 (それぞれ長さ Nx, Ny, Nz の 1D ベクトルのみ: 3D 配列を作らない)
    damp_x = jnp.asarray(
        _build_pure_jax_pml_profile(nx, pml_cells, min_is_pml=(px == 0), max_is_pml=True), dtype=dtype_jnp
    )[None, :, None, None]
    damp_y = jnp.asarray(
        _build_pure_jax_pml_profile(ny, pml_cells, min_is_pml=(py == 0), max_is_pml=True), dtype=dtype_jnp
    )[None, None, :, None]
    damp_z = jnp.asarray(
        _build_pure_jax_pml_profile(nz, pml_cells, min_is_pml=(pz == 0), max_is_pml=True), dtype=dtype_jnp
    )[None, None, None, :]

    def _sample_monitors(E_arr: jnp.ndarray, H_arr: jnp.ndarray) -> jnp.ndarray:
        """点モニタ群 (n_mon_pts, n_mon_comps) の現在値を抽出する。"""
        cols = []
        for eh_flag, c_idx in zip(mon_eh_flags, mon_comp_indices):
            field_arr = E_arr if eh_flag == 0 else H_arr
            vals = field_arr[c_idx, mon_ix, mon_iy, mon_iz]
            cols.append(vals.astype(jnp.float32))
        return jnp.stack(cols, axis=-1)

    def _extract_2d_slices(E_arr: jnp.ndarray, H_arr: jnp.ndarray):
        """STEP 3 用の 4 断面 (xy_z0, xy_zoff, xz_y0, yz_x0) の電磁場 (6, N1, N2) を抽出する。"""
        eh_z0 = jnp.concatenate([E_arr[:, :, :, iz_z0], H_arr[:, :, :, iz_z0]], axis=0).astype(jnp.complex64)
        eh_zoff = jnp.concatenate([E_arr[:, :, :, iz_zoff], H_arr[:, :, :, iz_zoff]], axis=0).astype(jnp.complex64)
        eh_y0 = jnp.concatenate([E_arr[:, :, iy_y0, :], H_arr[:, :, iy_y0, :]], axis=0).astype(jnp.complex64)
        eh_x0 = jnp.concatenate([E_arr[:, ix_x0, :, :], H_arr[:, ix_x0, :, :]], axis=0).astype(jnp.complex64)
        return eh_z0, eh_zoff, eh_y0, eh_x0

    def _single_fdtd_step(step_i: jnp.ndarray, state_carry, inv_eps_const: jnp.ndarray):
        """1 タイムステップ分の FDTD 更新と双極子励振・鏡像対称境界適用。"""
        t_cur = step_i.astype(jnp.float32) * float(dt_meep)
        env = jnp.exp(-0.5 * ((t_cur - float(t0_pulse)) / float(pulse_w)) ** 2) * jnp.where(
            t_cur <= float(t_src_end), 1.0, 0.0
        )

        if use_fdtdx_kernel:
            arrs = state_carry
            arrs = fdtdx_update_E(step_i, arrs, objects, sim_cfg, simulate_boundaries=True)
            E_cur = arrs.E
            for comp_i, ix, iy, iz, phi in e_src_list:
                val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                E_cur = E_cur.at[comp_i, ix, iy, iz].add(val)
            E_cur = apply_e_mirror_bc(E_cur, px=px, py=py, pz=pz)
            arrs = arrs.aset("E", E_cur)

            arrs = fdtdx_update_H(step_i, arrs, objects, sim_cfg, simulate_boundaries=True)
            H_cur = arrs.H
            for comp_i, ix, iy, iz, phi in h_src_list:
                val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                H_cur = H_cur.at[comp_i, ix, iy, iz].add(val)
            H_cur = apply_h_mirror_bc(H_cur, px=px, py=py, pz=pz)
            arrs = arrs.aset("H", H_cur)
            return arrs, arrs.E, arrs.H
        else:
            E_cur, H_cur = state_carry
            c_dt = jnp.asarray(courant_num, dtype=dtype_jnp)

            # 3D Yee curl_H (後方差分) - 中間テンソル寿命を最小化
            curl_hx = (H_cur[2] - jnp.roll(H_cur[2], 1, axis=1)) - (H_cur[1] - jnp.roll(H_cur[1], 1, axis=2))
            curl_hy = (H_cur[0] - jnp.roll(H_cur[0], 1, axis=2)) - (H_cur[2] - jnp.roll(H_cur[2], 1, axis=0))
            curl_hz = (H_cur[1] - jnp.roll(H_cur[1], 1, axis=0)) - (H_cur[0] - jnp.roll(H_cur[0], 1, axis=1))
            curl_h = jnp.stack([curl_hx, curl_hy, curl_hz], axis=0)

            E_cur = ((E_cur + c_dt * curl_h * inv_eps_const) * damp_x * damp_y) * damp_z
            for comp_i, ix, iy, iz, phi in e_src_list:
                val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                E_cur = E_cur.at[comp_i, ix, iy, iz].add(val)
            E_cur = apply_e_mirror_bc(E_cur, px=px, py=py, pz=pz)

            # 3D Yee curl_E (前進差分)
            curl_ex = (jnp.roll(E_cur[2], -1, axis=1) - E_cur[2]) - (jnp.roll(E_cur[1], -1, axis=2) - E_cur[1])
            curl_ey = (jnp.roll(E_cur[0], -1, axis=2) - E_cur[0]) - (jnp.roll(E_cur[2], -1, axis=0) - E_cur[2])
            curl_ez = (jnp.roll(E_cur[1], -1, axis=0) - E_cur[1]) - (jnp.roll(E_cur[0], -1, axis=1) - E_cur[0])
            curl_e = jnp.stack([curl_ex, curl_ey, curl_ez], axis=0)

            H_cur = ((H_cur - c_dt * curl_e) * damp_x * damp_y) * damp_z
            for comp_i, ix, iy, iz, phi in h_src_list:
                val = (env * jnp.sin(2.0 * math.pi * float(fcen) * t_cur + phi)).astype(dtype_jnp)
                H_cur = H_cur.at[comp_i, ix, iy, iz].add(val)
            H_cur = apply_h_mirror_bc(H_cur, px=px, py=py, pz=pz)
            return (E_cur, H_cur), E_cur, H_cur

    # donate_argnums=(1, 2) により carry_state (E, H) と dft_acc のバッファをインプレース再利用し、
    # チャンク境界での 2倍メモリ確保を防止する
    @jax.jit(donate_argnums=(1, 2))
    def _run_chunk(start_step: jnp.ndarray, carry_state, dft_acc, inv_eps_const: jnp.ndarray):
        def _outer_step(c_tuple, sample_idx):
            st, dft_buffers = c_tuple
            base_step = start_step + sample_idx * stride

            def _inner_step(st_in, sub_k):
                st_out, _, _ = _single_fdtd_step(base_step + sub_k, st_in, inv_eps_const)
                return st_out, None

            st_after, _ = jax.lax.scan(_inner_step, st, jnp.arange(stride, dtype=jnp.int32))
            if use_fdtdx_kernel:
                E_now, H_now = st_after.E, st_after.H
            else:
                E_now, H_now = st_after

            mon_sample = _sample_monitors(E_now, H_now)

            if has_dft:
                t_now = (base_step + stride - 1).astype(jnp.float32) * float(dt_meep)
                active_mask = jnp.where(t_now > float(t_src_end), 1.0, 0.0)
                phases = jnp.exp(-1j * 2.0 * math.pi * dft_freqs_j * t_now) * active_mask
                eh_z0, eh_zoff, eh_y0, eh_x0 = _extract_2d_slices(E_now, H_now)
                b_z0, b_zoff, b_y0, b_x0 = dft_buffers
                b_z0 = b_z0 + phases[:, None, None, None] * eh_z0[None, ...]
                b_zoff = b_zoff + phases[:, None, None, None] * eh_zoff[None, ...]
                b_y0 = b_y0 + phases[:, None, None, None] * eh_y0[None, ...]
                b_x0 = b_x0 + phases[:, None, None, None] * eh_x0[None, ...]
                dft_buffers = (b_z0, b_zoff, b_y0, b_x0)

            return (st_after, dft_buffers), mon_sample

        (final_st, final_dft), chunk_mon_traces = jax.lax.scan(
            _outer_step,
            (carry_state, dft_acc),
            jnp.arange(samples_per_chunk, dtype=jnp.int32),
        )
        return final_st, final_dft, chunk_mon_traces

    # 初期状態の構築
    if use_fdtdx_kernel:
        carry = arrays_or_inveps
    else:
        e_init = jnp.zeros((3, nx, ny, nz), dtype=dtype_jnp)
        h_init = jnp.zeros((3, nx, ny, nz), dtype=dtype_jnp)
        carry = (e_init, h_init)

    if has_dft:
        dft_acc = (
            jnp.zeros((n_freqs, 6, nx, ny), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, nx, ny), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, nx, nz), dtype=jnp.complex64),
            jnp.zeros((n_freqs, 6, ny, nz), dtype=jnp.complex64),
        )
    else:
        dft_acc = ()

    all_traces: List[np.ndarray] = []
    peak_energy = 0.0
    final_energy = 0.0
    stopped_early = False
    current_step = 0

    t_start_wall = time.perf_counter()
    jit_compile_sec = 0.0

    try:
        while current_step < total_steps_max:
            t_chunk0 = time.perf_counter()
            carry, dft_acc, traces_dev = _run_chunk(
                jnp.asarray(current_step, dtype=jnp.int32), carry, dft_acc, inv_eps_dev
            )
            traces_np = np.asarray(traces_dev, dtype=np.float32)
            del traces_dev
            t_chunk1 = time.perf_counter()

            if current_step == 0:
                jit_compile_sec = t_chunk1 - t_chunk0

            all_traces.append(traces_np)
            current_step += chunk_steps
            t_cur_meep = current_step * dt_meep

            tail_len = max(1, samples_per_chunk // 4)
            chunk_energy = float(np.mean(traces_np[-tail_len:] ** 2))
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
            b_z0_np = np.asarray(dft_acc[0])
            b_zoff_np = np.asarray(dft_acc[1])
            b_y0_np = np.asarray(dft_acc[2])
            b_x0_np = np.asarray(dft_acc[3])

            eps_np = eps_np_full[0]
            eps_z0 = eps_np[:, :, iz_z0]
            eps_zoff = eps_np[:, :, iz_zoff]
            eps_y0 = eps_np[:, iy_y0, :]
            eps_x0 = eps_np[ix_x0, :, :]

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
        # GPU メモリバッファと JIT コンパイル済みグラフを即座に解放
        del carry, dft_acc, inv_eps_dev, arrays_or_inveps, _run_chunk
        clear_jit_cache()

    return FDTDXSimulationResult(
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
    )
