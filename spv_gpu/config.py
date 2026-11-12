"""
Configuration & automatic parameter derivation for spv_gpu (Spec A §4-5 & Spec B1 §4-5).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple
import numpy as np


# 構造タイプ別の既定パラメータ (spv/run_sim.py および spv/my_geometry.py と完全一致)
DEFAULT_STRUCTURES: Dict[int, Dict[str, Any]] = {
    6: {
        "n": 3.4,
        "a": 0.45,
        "period": 0.45,
        "r_ratio": 0.26,
        "slab_ratio": 0.50,
        "N_ring": 8,
        "twist_deg": 3.5,
        "core_r_ratio": 0.55,
        "gap_ratio": 0.02,
        "bowtie_angle": 60.0,
    },
    3: {
        "n": 2.4,
        "period": 1.5,
        "Disk_R": 1.0,
        "disk_D": 1.5,
        "w": 0.3,
        "Disk_t": 0.14,
        "disk_t": 0.14,
        "Sac_t": 0.5,
        "Sac_D": 0.0,
        "Out_D": 10.0,
        "Gap": 0.12,
        "Wg_w": 0.155,
        "Wg_t": 0.14,
        "Cir_reso": 200,
        "Dev_rate": 0.0,
    },
    4: {
        "n": 2.4,
        "period": 1.5,
        "Disk_R": 1.0,
        "disk_D": 1.5,
        "w": 0.3,
        "Disk_t": 0.14,
        "disk_t": 0.14,
        "Sac_t": 0.5,
        "Sac_D": 0.0,
        "Out_D": 10.0,
        "Gap": 0.12,
        "Wg_w": 0.155,
        "Wg_t": 0.14,
        "Cir_reso": 200,
        "Dev_rate": 0.0,
    },
    5: {
        "n": 2.4,
        "period": 1.5,
        "Disk_R": 1.0,
        "disk_D": 1.5,
        "w": 0.3,
        "Disk_t": 0.14,
        "disk_t": 0.14,
        "Sac_t": 0.5,
        "Sac_D": 0.0,
        "Out_D": 10.0,
        "Gap": 0.12,
        " gap": 0.12,
        "Wg_w": 0.155,
        "wg_w": 0.155,
        "Wg_t": 0.14,
        "wg_t": 0.14,
        "Cir_reso": 200,
        "Dev_rate": 0.0,
    },
    0: {
        "n": 3.4,
        "period": 0.42,
        "r_ratio": 0.35,
        "slab_ratio": 0.40,
        "Nx": 6,
        "Ny": 5,
        "defN": 1,
    },
}


DEFAULT_CFG: Dict[str, Any] = {
    # --- 仕様書A §4 / 仕様書B1 §4 共通ユーザー入力 ---
    "structure_type": 6,  # 0:2D_PhC(V2検証用), 3:microdisk, 4:rough_microdisk, 5:waveguide_disk, 6:singular_bowtie_phc
    "structure": {
        "n": 3.4,
        "a": 0.45,
        "r_ratio": 0.26,
        "slab_ratio": 0.50,
        "N_ring": 8,
        "twist_deg": 3.5,
        "core_r_ratio": 0.55,
        "gap_ratio": 0.02,
        "bowtie_angle": 60.0,
    },
    "f_range": (0.55, 0.75),
    "parities": ["+", "-"],
    "accuracy": {"coarse": 2, "fine": 4},
    "n_modes": 2,
    "run_mode": "ALL",  # 'ALL' | 'HARMINV' | 'MODE'
    # --- 後方互換・詳細制御キー ---
    "n_core": 3.4,
    "n_clad": 1.0,
    "n_sac": 3.4,  # spv/my_geometry.py では犠牲層・基板も同一 medium (n)
    "period": 0.45,
    "geom": {
        # type 6: Singular Bowtie PhC (spv/run_sim.py 既定値)
        "r_ratio": 0.26,
        "slab_ratio": 0.50,
        "N_ring": 8,
        "twist_deg": 3.5,
        "core_r_ratio": 0.55,
        "gap_ratio": 0.02,
        "bowtie_angle": 60.0,
        # type 3, 4, 5: Microdisk / Rough disk / Waveguide disk (spv/run_sim.py 既定値)
        "Disk_R": 1.0,
        "w": 0.3,
        "disk_D": 1.5,
        "disk_t": 0.14,
        "Disk_t": 0.14,
        "Sac_D": 0.0,
        "Sac_t": 0.5,
        "Out_D": 10.0,
        "Cir_reso": 200,
        "Dev_rate": 0.0,
        "wg_w": 0.155,
        "Wg_w": 0.155,
        "wg_t": 0.14,
        "Wg_t": 0.14,
        "gap": 0.12,
        "Gap": 0.12,
        # type 0: 2D PhC (V2 検証用)
        "Nx": 6,
        "Ny": 5,
        "defN": 1,
        "Num_air": 8,
    },
    "symmetry": {"px": 0, "py": 0, "pz": "auto"},
    "quality": "standard",
    "n_modes_select": 2,
    "Q_expected": 1e4,
    "merge_refine_mode": True,
    "output_dir": "/content/drive/MyDrive/spv_results",
    "random_seed": 42,
}

DEFAULT_BACKEND: Dict[str, Any] = {
    "dtype": "float32",  # 'float32' (既定) | 'bfloat16'
    "max_gpu_mem_frac": 0.70,  # BFCアロケータの断片化・XLA中間バッファ余力を確保するため既定70%
    "x64_refine": False,  # 高Qモードの補正だけ float64 を選べるようにする
    "chunk_steps": 1000,  # lax.scan の1チャンクあたりステップ数
    "subpixel_n": 4,  # 4x4x4 サブピクセル体積分率平均
    "anisotropic_averaging": False,
}


def normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    仕様書A §4 形式 (`structure`, `parities`, `accuracy`, `n_modes`) と
    既存の `geom`, `n_core`, `symmetry`, `n_modes_select` の双方向互換を正規化する。
    """
    c = copy.deepcopy(DEFAULT_CFG)
    c.update(copy.deepcopy(cfg))

    stype = int(c.get("structure_type", 6))
    base_st = copy.deepcopy(DEFAULT_STRUCTURES.get(stype, DEFAULT_STRUCTURES[6]))

    # geom と structure の両方をマージ（ユーザー指定のキーを優先）
    user_geom = cfg.get("geom", {})
    user_struct = cfg.get("structure", {})
    merged_st = copy.deepcopy(base_st)
    merged_st.update(user_geom)
    merged_st.update(user_struct)

    # キー別名の同期 (Disk_t <-> disk_t, Wg_w <-> wg_w, Gap <-> gap, a <-> period)
    if "a" in user_struct and "period" not in user_struct:
        merged_st["period"] = float(user_struct["a"])
    elif "period" in user_struct and "a" not in user_struct:
        merged_st["a"] = float(user_struct["period"])
    elif "period" in cfg and "a" not in user_struct and "period" not in user_struct:
        merged_st["period"] = float(cfg["period"])
        merged_st["a"] = float(cfg["period"])

    if "Disk_t" in user_struct or "Disk_t" in user_geom:
        merged_st["disk_t"] = float(merged_st["Disk_t"])
    elif "disk_t" in user_struct or "disk_t" in user_geom:
        merged_st["Disk_t"] = float(merged_st["disk_t"])

    if "Wg_w" in user_struct or "Wg_w" in user_geom:
        merged_st["wg_w"] = float(merged_st["Wg_w"])
    elif "wg_w" in user_struct or "wg_w" in user_geom:
        merged_st["Wg_w"] = float(merged_st["wg_w"])

    if "Wg_t" in user_struct or "Wg_t" in user_geom:
        merged_st["wg_t"] = float(merged_st["Wg_t"])
    elif "wg_t" in user_struct or "wg_t" in user_geom:
        merged_st["Wg_t"] = float(merged_st["wg_t"])

    if "Gap" in user_struct or "Gap" in user_geom:
        merged_st["gap"] = float(merged_st["Gap"])
    elif "gap" in user_struct or "gap" in user_geom:
        merged_st["Gap"] = float(merged_st["gap"])

    if "disk_D" not in user_struct and "disk_D" not in user_geom:
        merged_st["disk_D"] = float(merged_st["period"])

    # 屈折率 n の同期 (spv/run_sim.py では type 6 は n=3.4、type 3〜5 は n=2.4)
    if "n" in user_struct or "n" in user_geom:
        n_val = float(merged_st["n"])
    elif "n_core" in cfg:
        n_val = float(cfg["n_core"])
    else:
        n_val = float(base_st["n"])

    merged_st["n"] = n_val
    c["n_core"] = n_val
    c["n_sac"] = float(cfg.get("n_sac", n_val))
    c["period"] = float(merged_st["period"])
    c["structure"] = merged_st
    c["geom"] = merged_st

    if "n_modes" in cfg and "n_modes_select" not in cfg:
        c["n_modes_select"] = int(cfg["n_modes"])
    elif "n_modes_select" in cfg and "n_modes" not in cfg:
        c["n_modes"] = int(cfg["n_modes_select"])

    return c


def get_default_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """デフォルトの cfg と backend 設定のディープコピーを返す。"""
    return normalize_cfg(DEFAULT_CFG), copy.deepcopy(DEFAULT_BACKEND)


def min_feature(cfg: Dict[str, Any]) -> float:
    """
    仕様書A §5 に基づく構造別の最小特徴寸法 d_min [um] を返す。
    - type 6: gap_ratio * a
    - type 5: min(Gap, Wg_w, Wg_t, Disk_t)
    - type 3/4: min(w, Disk_t)
    """
    c = normalize_cfg(cfg)
    stype = int(c["structure_type"])
    a = float(c["period"])
    g = c["structure"]

    if stype == 6:
        return float(g.get("gap_ratio", 0.02)) * a
    elif stype == 5:
        return float(min(g.get("Gap", 0.12), g.get("Wg_w", 0.155), g.get("Wg_t", 0.14), g.get("Disk_t", 0.14)))
    elif stype in (3, 4):
        return float(min(g.get("w", 0.3), g.get("Disk_t", 0.14)))
    elif stype == 0:
        slab_t = float(g.get("slab_ratio", 0.40)) * a
        hole_d = 2.0 * float(g.get("r_ratio", 0.35)) * a
        return float(min(slab_t, hole_d))
    else:
        return max(a * 0.1, 0.05)


def estimate_gpu_memory_gb(
    nx: int, ny: int, nz: int, dtype: str = "float32", n_dft_freqs: int = 0
) -> float:
    """
    グリッドサイズ (Nx, Ny, Nz) から FDTDX + XLA lax.scan 実行時のピーク GPU メモリ使用量 [GB] を算出する。
    E(3), H(3), inv_eps(1-3), PML補助場 psi(18), curl/roll中間テンソルおよびXLAダブルバッファを考慮し、
    1セルあたり 64 要素分で安全側に見積もる。
    """
    bytes_per_elem = 2 if dtype == "bfloat16" else (8 if dtype == "float64" else 4)
    base_bytes = float(nx) * float(ny) * float(nz) * bytes_per_elem * 64.0
    slice_elems = (nx * ny * 2 + nx * nz + ny * nz) * 6 * max(1, n_dft_freqs)
    dft_bytes = slice_elems * 8.0 * 2.0
    return (base_bytes + dft_bytes) / (1024.0**3)


def detect_available_gpu_memory_gb() -> float:
    """利用可能な GPU の総 VRAM [GB] を検出する。未検出時は Colab T4 相当の 15.0 GB を返す。"""
    try:
        import jax

        devs = jax.devices()
        for d in devs:
            if d.platform == "gpu":
                stats = d.memory_stats()
                if stats and "bytes_limit" in stats:
                    return float(stats["bytes_limit"]) / (1024.0**3)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        mb = float(out.strip().splitlines()[0])
        return mb / 1024.0
    except Exception:
        return 15.0


def _compute_structure_bounding_box(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """構造本体の外接サイズ (Lx_struct, Ly_struct, Lz_struct) [um] を返す (spv/run_sim.py 準拠)。"""
    c = normalize_cfg(cfg)
    stype = int(c["structure_type"])
    a = float(c["period"])
    g = c["structure"]

    if stype == 6:
        n_ring = int(g.get("N_ring", 8))
        slab_t = round(float(g.get("slab_ratio", 0.50)) * a, 3)
        # spv/run_sim.py L228: xy_span = 2 * N_ring * period + 2 * dist_to_pml[0]
        span_xy = 2.0 * n_ring * a
        return (span_xy, span_xy, slab_t)
    elif stype in (3, 4, 5):
        disk_d = float(g.get("disk_D", a))
        disk_t = float(g.get("Disk_t", 0.14))
        return (disk_d, disk_d, disk_t)
    elif stype == 0:
        nx_p = int(g.get("Nx", 6))
        ny_p = int(g.get("Ny", 5))
        slab_t = round(float(g.get("slab_ratio", 0.40)) * a, 3)
        return (2.0 * nx_p * a, 2.0 * ny_p * (math.sqrt(3.0) / 2.0) * a, slab_t)
    else:
        return (4.0 * a, 4.0 * a, 0.5 * a)


def _generate_default_sources_and_monitors(
    cfg: Dict[str, Any],
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    仕様書A §5 に基づく同時励起波源位置（ギャップ中心 + 0.05a オフセット点）および
    構造別の既定監視点リスト (実座標 [um]) を生成する。
    """
    c = normalize_cfg(cfg)
    stype = int(c["structure_type"])
    a = float(c["period"])
    g = c["structure"]

    # 仕様書A §5: ギャップ（構造の中心）と、そこから 0.05a ずらした点の2か所
    src_positions = [
        (0.0, 0.0, 0.0),
        (0.05 * a, 0.05 * a, 0.0),
    ]

    if stype == 6:
        slab_ratio = float(g.get("slab_ratio", 0.50))
        m_ratios = [
            (0.0, 0.0, 0.0),
            (0.02, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (0.10, 0.0, 0.0),
            (0.0, 0.05, 0.0),
            (0.05, 0.05, 0.0),
            (0.15, 0.10, 0.0),
            (0.0, 0.0, 0.25 * slab_ratio),
        ]
        mon_positions = [(rx * a, ry * a, rz * a) for rx, ry, rz in m_ratios]
    elif stype in (3, 4, 5):
        disk_t = float(g.get("Disk_t", 0.14))
        ratio = disk_t / a
        m_ratios = [
            (0.0, 0.475, 0.0 * ratio),
            (0.0, 0.475, 0.45 * ratio),
            (ratio / 2.0, 0.475, 0.0 * ratio),
            (ratio / 2.0, 0.475, 0.45 * ratio),
            (ratio, 0.475, 0.0 * ratio),
            (ratio, 0.475, 0.45 * ratio),
            (0.475, 0.0, 0.0 * ratio),
            (0.475, 0.0, 0.45 * ratio),
            (0.475, ratio / 2.0, 0.0 * ratio),
            (0.475, ratio / 2.0, 0.45 * ratio),
            (0.475, ratio, 0.0 * ratio),
            (0.475, ratio, 0.45 * ratio),
        ]
        mon_positions = [(rx * a, ry * a, rz * a) for rx, ry, rz in m_ratios]
    else:
        mon_positions = [
            (0.0, 0.0, 0.0),
            (0.10 * a, 0.05 * a, 0.0),
            (0.20 * a, 0.10 * a, 0.0),
        ]
    return src_positions, mon_positions


def derive_parameters(
    cfg: Dict[str, Any],
    backend: Dict[str, Any] | None = None,
    gpu_mem_gb: float | None = None,
) -> Dict[str, Any]:
    """
    仕様書A §5 および 仕様書B1 §5 に従い、ユーザ入力 cfg から全計算パラメータを自動導出する。
    """
    if backend is None:
        backend = DEFAULT_BACKEND
    if gpu_mem_gb is None:
        gpu_mem_gb = detect_available_gpu_memory_gb()

    c = normalize_cfg(cfg)
    stype = int(c["structure_type"])
    a = float(c["period"])
    n_core = float(c["n_core"])
    n_clad = float(c.get("n_clad", 1.0))

    fmin, fmax = float(c["f_range"][0]), float(c["f_range"][1])
    # 仕様書A §5: fcen = (f_min + f_max)/2, df = 1.2 * (f_max - f_min), cutoff = 5.0
    fcen = 0.5 * (fmin + fmax)
    df = 1.2 * (fmax - fmin)
    cutoff = 5.0

    # 仕様書A §5 & spv/run_sim.py:
    # spv/run_sim.py では Meep の長さ単位 = 1 um であり、res_harminv / res_mode は既に [pixels/um] 単位。
    # type 6 の微小ギャップ (gap_ratio * a = 9nm) は 4x4x4 サブピクセル平滑化で表現し、
    # ベースグリッド解像度は spv/run_sim.py 既定値 (type 6: 55 / 80, type 3-5: 60 / 90) を基準とする。
    d_min = min_feature(c)
    res_floor = math.ceil(12.0 * n_core * fmax)

    quality = c.get("quality", "standard")
    if "accuracy" in cfg and isinstance(cfg["accuracy"], dict):
        acc_coarse = float(cfg["accuracy"].get("coarse", 2.0))
        acc_fine = float(cfg["accuracy"].get("fine", 4.0))
    else:
        acc_coarse, acc_fine = (1.5, 3.0) if quality == "fast" else ((2.5, 4.5) if quality == "high" else (2.0, 4.0))

    if stype == 6:
        # spv/run_sim.py L214-215: coarse=2 -> 55 px/um, fine=4 -> 80 px/um
        res_harminv = max(int(round(27.5 * acc_coarse)), res_floor)
        res_mode = max(int(round(20.0 * acc_fine)), res_floor)
    elif stype in (3, 4, 5):
        # spv/run_sim.py L284-285: coarse=2 -> 60 px/um, fine=4 -> 90 px/um
        res_harminv = max(int(round(30.0 * acc_coarse)), res_floor)
        res_mode = max(int(round(22.5 * acc_fine)), res_floor)
    else:
        res_harminv = max(int(round(25.0 * acc_coarse)), res_floor)
        res_mode = max(int(round(20.0 * acc_fine)), res_floor)

    res_harminv = min(res_harminv, 90)
    res_mode = min(res_mode, 100)

    if "res_harminv" in cfg and cfg["res_harminv"] is not None:
        res_harminv = int(cfg["res_harminv"])
    if "res_mode" in cfg and cfg["res_mode"] is not None:
        res_mode = int(cfg["res_mode"])

    res_harminv_meep = res_harminv
    res_mode_meep = res_mode

    # §5 dist_to_pml と pml_w (spv/run_sim.py L225-230, L287-289 と完全一致)
    if stype == 6:
        dist_xy = 0.35
        dist_z = 0.55
        pml_w_base = 0.35
    else:
        dist_xy = 0.70
        dist_z = 0.60
        pml_w_base = 0.10

    lx_s, ly_s, lz_s = _compute_structure_bounding_box(c)

    def _calc_cell_and_grid(res_um: int, pz_val: int) -> Tuple[Tuple[float, float, float], float, Tuple[int, int, int]]:
        pml_cells = max(10, int(math.ceil(pml_w_base * res_um)))
        pml_w = pml_cells / float(res_um)
        if stype == 6:
            # spv/run_sim.py L228-230 + my_mp_setup.py L40-42: sx = xm + 2*pml_w
            sx = lx_s + 2.0 * (dist_xy + pml_w)
            sy = ly_s + 2.0 * (dist_xy + pml_w)
            sz = lz_s + 2.0 * (dist_z + pml_w)
        else:
            sx = lx_s + dist_xy + 2.0 * pml_w
            sy = ly_s + dist_xy + 2.0 * pml_w
            sz = lz_s + dist_z + 2.0 * pml_w
        nx = max(16, int(round(sx * res_um)))
        ny = max(16, int(round(sy * res_um)))
        nz_full = max(16, int(round(sz * res_um)))
        nx += nx % 2
        ny += ny % 2
        nz_full += nz_full % 2
        nz = (nz_full // 2) if pz_val != 0 else nz_full
        return (nx / float(res_um), ny / float(res_um), nz_full / float(res_um)), pml_w, (nx, ny, nz)

    # 対称性とパリティ展開 (仕様書A §4 parities=['+', '-'] および symmetry 辞書の両方に対応)
    sym_cfg = c.get("symmetry", {"px": 0, "py": 0, "pz": "auto"})
    px = int(sym_cfg.get("px", 0))
    py = int(sym_cfg.get("py", 0))

    p_map = {"+": 1, "-": -1, "o": 0, 1: 1, -1: -1, 0: 0}
    if "parities" in cfg and cfg["parities"] is not None:
        parities = [p_map[p] for p in cfg["parities"]]
    else:
        pz = sym_cfg.get("pz", "auto")
        if pz == "auto":
            parities = [+1, -1]
        elif isinstance(pz, (list, tuple)):
            parities = [p_map[p] for p in pz]
        else:
            parities = [p_map[pz]]

    # GPUメモリ上限チェック (仕様書B1 §5)
    max_allowed_gb = float(backend.get("max_gpu_mem_frac", 0.9)) * gpu_mem_gb
    dtype_str = backend.get("dtype", "float32")
    pz_check = parities[0] if parities else 0

    while res_harminv > 16:
        _, _, (nx, ny, nz) = _calc_cell_and_grid(res_harminv, pz_check)
        if estimate_gpu_memory_gb(nx, ny, nz, dtype=dtype_str) <= max_allowed_gb:
            break
        res_harminv -= 2

    while res_mode > 20:
        _, _, (nx, ny, nz) = _calc_cell_and_grid(res_mode, pz_check)
        if estimate_gpu_memory_gb(nx, ny, nz, dtype=dtype_str, n_dft_freqs=c.get("n_modes_select", 2)) <= max_allowed_gb:
            break
        res_mode -= 2

    sp_harminv, pml_w_harminv, grid_harminv = _calc_cell_and_grid(res_harminv, pz_check)
    sp_mode, pml_w_mode, grid_mode = _calc_cell_and_grid(res_mode, pz_check)

    # パリティ毎の監視・励起成分 (仕様書A §5: 偶 ex,ey,hz / 奇 hx,hy,ez)
    parity_components: Dict[int, Dict[str, List[str]]] = {
        +1: {"src_comps": ["Ex", "Ey", "Hz"], "mon_comps": ["Ex", "Ey", "Hz"]},
        -1: {"src_comps": ["Hx", "Hy", "Ez"], "mon_comps": ["Hx", "Hy", "Ez"]},
        0: {"src_comps": ["Ex", "Ey", "Ez", "Hz"], "mon_comps": ["Ex", "Ey", "Ez", "Hz"]},
    }

    src_positions, mon_positions = _generate_default_sources_and_monitors(c)
    if "src_positions" in cfg and cfg["src_positions"]:
        src_positions = list(cfg["src_positions"])
    if "mon_positions" in cfg and cfg["mon_positions"]:
        mon_positions = list(cfg["mon_positions"])

    rng = np.random.default_rng(int(c.get("random_seed", 42)))
    src_phases = rng.uniform(0.0, 2.0 * math.pi, size=(len(src_positions), 6))

    # 時間発展・減衰停止条件
    # ※ FDTDX カーネル内の周波数は um^-1 単位または Meep a/lambda 単位
    t_src_end = 2.0 * cutoff / df
    q_exp = float(c.get("Q_expected", 1e4))
    t_max_harminv = t_src_end + min(2000.0, max(300.0, q_exp / max(fcen, 0.1)))
    t_max_refine = t_src_end + min(2000.0, max(450.0, 5.0 * q_exp / max(fcen, 0.1)))
    t_max_mode = 2.0 * cutoff / max(df * 0.05, fcen / 500.0) + 200.0 / max(fcen, 0.1)

    courant_number = 0.99 / math.sqrt(3.0)
    dt_harminv = courant_number / float(res_harminv)
    dt_mode = courant_number / float(res_mode)

    stride_harminv = max(1, int(math.floor(1.0 / (8.0 * dt_harminv * fmax))))
    stride_mode = max(1, int(math.floor(1.0 / (8.0 * dt_mode * fmax))))

    lam_min_mat = 1.0 / (fmax * n_core)
    lam_max_clad = 1.0 / (fmin * n_clad)

    derived = {
        "fcen": fcen,
        "df": df,
        "fmin": fmin,
        "fmax": fmax,
        "cutoff": cutoff,
        "lam_min_mat": lam_min_mat,
        "lam_max_clad": lam_max_clad,
        "feat_min": d_min,
        "d_min": d_min,
        "res_harminv": res_harminv,
        "res_mode": res_mode,
        "res_harminv_meep": res_harminv_meep,
        "res_mode_meep": res_mode_meep,
        "dist_to_pml": [dist_xy, dist_xy, dist_z],
        "pml_w_harminv": pml_w_harminv,
        "pml_w_mode": pml_w_mode,
        "pml_cells_harminv": max(10, int(round(pml_w_harminv * res_harminv))),
        "pml_cells_mode": max(10, int(round(pml_w_mode * res_mode))),
        "sp_harminv": sp_harminv,
        "sp_mode": sp_mode,
        "grid_harminv": grid_harminv,
        "grid_mode": grid_mode,
        "px": px,
        "py": py,
        "parities": parities,
        "parity_components": parity_components,
        "src_positions": src_positions,
        "mon_positions": mon_positions,
        "src_phases": src_phases,
        "t_src_end": t_src_end,
        "T_MAX_harminv": t_max_harminv,
        "T_MAX_refine": t_max_refine,
        "T_MAX_mode": t_max_mode,
        "dt_harminv": dt_harminv,
        "dt_mode": dt_mode,
        "stride_harminv": stride_harminv,
        "stride_mode": stride_mode,
        "est_mem_harminv_gb": estimate_gpu_memory_gb(*grid_harminv, dtype=dtype_str),
        "est_mem_mode_gb": estimate_gpu_memory_gb(
            *grid_mode, dtype=dtype_str, n_dft_freqs=c.get("n_modes_select", 2)
        ),
        "gpu_mem_gb": gpu_mem_gb,
    }
    return derived
