"""
Configuration & automatic parameter derivation for spv_colab_sync.
Aligned with spv/run_sim.py and spv_gpu/config.py (resolution, PML, cell size,
z-asymmetry handling for type 3/4, time limits, and unified summary.csv schema).
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Tuple
import numpy as np


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
        "Wg_w": 0.155,
        "Wg_t": 0.14,
        "Cir_reso": 200,
        "Dev_rate": 0.0,
    },
    0: {
        "n": 3.4,
        "period": 0.42,
        "a": 0.42,
        "r_ratio": 0.35,
        "slab_ratio": 0.40,
        "Nx": 6,
        "Ny": 5,
        "defN": 1,
    },
}


DEFAULT_CFG: Dict[str, Any] = {
    "structure_type": 6,
    "structure": copy.deepcopy(DEFAULT_STRUCTURES[6]),
    "f_range": (0.55, 0.75),
    "parities": ["+", "-"],
    "symmetry": {"px": 0, "py": 0, "pz": "auto"},
    "quality": "standard",  # "fast" | "standard" | "high"
    "accuracy": {"coarse": 2.0, "fine": 4.0},
    "n_modes": 2,
    "n_modes_select": 2,
    "Q_expected": 1e4,
    "run_mode": "ALL",  # "ALL" | "HARMINV" | "MODE"
    "merge_refine_mode": True,
    "use_decay_stop": True,
    "decay_dt": 50.0,
    "decay_by": 1e-5,
    "n_shots": 1,
    "random_seed": 42,
    "save_h5": False,
    "legacy_spv_cell": False,
    "res_harminv": None,
    "res_mode": None,
    "output_dir": os.environ.get("SPV_OUT_DIR", "./spv_colab_sync_results"),
}

cfg = copy.deepcopy(DEFAULT_CFG)

PARITY_MAP: Dict[Any, Any] = {"+": 1, "-": -1, "o": 0, 1: "+", -1: "-", 0: "o"}

# spv_gpu および spv_colab の双方と互換性を持つ共通 summary.csv スキーマ
SUMMARY_COLUMNS: List[str] = [
    "parity",
    "sym",
    "pz",
    "freq",
    "Q",
    "Q_max",
    "Q_iqr",
    "|A|",
    "Re[A]",
    "Im[A]",
    "Re[Er]",
    "n_detections",
    "rank_score",
    "res",
    "elapsed_sec",
    "excon",
    "detcon",
    "file name",
    "profile_png",
]

CUTOFF = 5.0
ATH = 1e-5
Q_MIN = 20.0
REFINE_MATCH_FACTOR = 0.6
Q_MERGE_THRESHOLD = 100.0


def get_default_config() -> Dict[str, Any]:
    """デフォルト設定辞書のディープコピーを返す。"""
    return normalize_cfg(DEFAULT_CFG)


def normalize_cfg(user_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    spv_gpu / spv_colab 両形式の入力キーを統一正規化する。
    """
    raw = user_cfg or {}
    c = copy.deepcopy(DEFAULT_CFG)
    c.update(copy.deepcopy(raw))

    stype = int(c.get("structure_type", 6))
    if stype not in DEFAULT_STRUCTURES:
        raise ValueError(f"未対応の structure_type です: {stype} (対応: 0, 3, 4, 5, 6)")

    base_st = copy.deepcopy(DEFAULT_STRUCTURES[stype])
    user_geom = raw.get("geom", {}) if isinstance(raw.get("geom"), dict) else {}
    user_struct = raw.get("structure", {}) if isinstance(raw.get("structure"), dict) else {}

    merged = copy.deepcopy(base_st)
    merged.update(user_geom)
    merged.update(user_struct)

    if "a" in user_struct and "period" not in user_struct:
        merged["period"] = float(user_struct["a"])
    elif "period" in user_struct and "a" not in user_struct:
        merged["a"] = float(user_struct["period"])
    elif "period" in raw and "a" not in user_struct and "period" not in user_struct:
        merged["period"] = float(raw["period"])
        merged["a"] = float(raw["period"])
    else:
        val_p = float(merged.get("period", merged.get("a", base_st.get("period", 0.45))))
        merged["period"] = val_p
        merged["a"] = val_p

    if "Disk_t" in user_struct or "Disk_t" in user_geom:
        merged["disk_t"] = float(merged["Disk_t"])
    elif "disk_t" in user_struct or "disk_t" in user_geom:
        merged["Disk_t"] = float(merged["disk_t"])

    if "Wg_w" in user_struct or "Wg_w" in user_geom:
        merged["wg_w"] = float(merged["Wg_w"])
    elif "wg_w" in user_struct or "wg_w" in user_geom:
        merged["Wg_w"] = float(merged["wg_w"])

    if "Wg_t" in user_struct or "Wg_t" in user_geom:
        merged["wg_t"] = float(merged["Wg_t"])
    elif "wg_t" in user_struct or "wg_t" in user_geom:
        merged["Wg_t"] = float(merged["wg_t"])

    if "Gap" in user_struct or "Gap" in user_geom:
        merged["gap"] = float(merged["Gap"])
    elif "gap" in user_struct or "gap" in user_geom:
        merged["Gap"] = float(merged["gap"])

    if "disk_D" not in user_struct and "disk_D" not in user_geom:
        merged["disk_D"] = float(merged["period"])

    if "n" in user_struct or "n" in user_geom:
        merged["n"] = float(merged["n"])
    elif "n_core" in raw:
        merged["n"] = float(raw["n_core"])

    c["structure"] = merged
    c["geom"] = merged
    c["period"] = float(merged["period"])
    c["n_core"] = float(merged["n"])

    if "n_modes_select" in raw and "n_modes" not in raw:
        c["n_modes"] = int(raw["n_modes_select"])
    elif "n_modes" in raw and "n_modes_select" not in raw:
        c["n_modes_select"] = int(raw["n_modes"])
    else:
        c["n_modes"] = int(c.get("n_modes", c.get("n_modes_select", 2)))
        c["n_modes_select"] = int(c["n_modes"])

    if "out_dir" in raw and "output_dir" not in raw:
        c["output_dir"] = str(raw["out_dir"])
    elif "output_dir" in raw and "out_dir" not in raw:
        c["out_dir"] = str(raw["output_dir"])
    else:
        c["out_dir"] = str(c.get("output_dir", os.environ.get("SPV_OUT_DIR", "./spv_colab_sync_results")))
        c["output_dir"] = c["out_dir"]

    if "seed" in raw and "random_seed" not in raw:
        c["random_seed"] = int(raw["seed"])

    return c


def min_feature(user_cfg: Dict[str, Any]) -> float:
    """構造別の最小特徴寸法 d_min [um] を返す。"""
    c = normalize_cfg(user_cfg)
    stype = int(c["structure_type"])
    st = c["structure"]
    a = float(st["period"])
    if stype == 6:
        return float(st["gap_ratio"]) * a
    elif stype == 5:
        return float(min(st["Gap"], st["Wg_w"], st["Wg_t"], st["Disk_t"]))
    elif stype in (3, 4):
        return float(min(st["w"], st["Disk_t"]))
    elif stype == 0:
        slab_t = float(st["slab_ratio"]) * a
        hole_d = 2.0 * float(st["r_ratio"]) * a
        return float(min(slab_t, hole_d))
    raise ValueError(f"未対応の structure_type です: {stype}")


def _compute_structure_bounding_box(c: Dict[str, Any]) -> Tuple[float, float, float]:
    """構造本体の外接サイズ (Lx, Ly, Lz) [um] を返す。"""
    stype = int(c["structure_type"])
    st = c["structure"]
    a = float(st["period"])
    if stype == 6:
        n_ring = int(st["N_ring"])
        slab_t = round(float(st["slab_ratio"]) * a, 3)
        span_xy = 2.0 * n_ring * a
        return (span_xy, span_xy, slab_t)
    elif stype in (3, 4, 5):
        disk_d = float(st.get("disk_D", a))
        disk_t = float(st["Disk_t"])
        return (disk_d, disk_d, disk_t)
    elif stype == 0:
        nx_p = int(st.get("Nx", 6))
        ny_p = int(st.get("Ny", 5))
        slab_t = round(float(st["slab_ratio"]) * a, 3)
        return (2.0 * nx_p * a, 2.0 * ny_p * (math.sqrt(3.0) / 2.0) * a, slab_t)
    raise ValueError(f"未対応の structure_type です: {stype}")


def get_monitor_fields(pz: Any) -> List[str]:
    """z 対称性に応じた監視・励起成分を返す。"""
    pz_val = PARITY_MAP[pz] if isinstance(pz, str) else int(pz)
    if pz_val == 1:
        return ["ex", "ey", "hz"]
    elif pz_val == -1:
        return ["hx", "hy", "ez"]
    return ["ex", "ey", "ez", "hx", "hy", "hz"]


def get_default_monitor_positions(c: Dict[str, Any]) -> List[List[float]]:
    """構造ごとの既定監視点リスト (period 比座標) を返す。"""
    stype = int(c["structure_type"])
    st = c["structure"]
    if stype == 6:
        slab_ratio = float(st["slab_ratio"])
        return [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.15, 0.10, 0.0],
            [0.0, 0.0, 0.25 * slab_ratio],
        ]
    elif stype in (3, 4, 5):
        disk_t = float(st["Disk_t"])
        period = float(st["period"])
        ratio = disk_t / period
        return [
            [0.0, 0.475, 0.0 * ratio],
            [0.0, 0.475, 0.45 * ratio],
            [ratio / 2.0, 0.475, 0.0 * ratio],
            [ratio / 2.0, 0.475, 0.45 * ratio],
            [ratio, 0.475, 0.0 * ratio],
            [ratio, 0.475, 0.45 * ratio],
            [0.475, 0.0, 0.0 * ratio],
            [0.475, 0.0, 0.45 * ratio],
            [0.475, ratio / 2.0, 0.0 * ratio],
            [0.475, ratio / 2.0, 0.45 * ratio],
            [0.475, ratio, 0.0 * ratio],
            [0.475, ratio, 0.45 * ratio],
        ]
    return [
        [0.0, 0.0, 0.0],
        [0.10, 0.05, 0.0],
        [0.20, 0.10, 0.0],
    ]


def get_symmetry_combinations(c: Dict[str, Any], pz_char: str) -> List[List[int]]:
    """
    構造ごとの (px, py, pz) 対称性リストを返す。
    type=3, 4 (microdisk / rough_microdisk) は z < 0 にのみ犠牲層と基板が存在する
    z 非対称構造であるため、spv/run_sim.py と同様に pz=0 を強制する。
    """
    stype = int(c["structure_type"])
    pz_val = PARITY_MAP[pz_char] if isinstance(pz_char, str) else int(pz_char)
    if stype in (3, 4):
        return [[-1, 1, 0], [1, 1, 0]]
    elif stype == 5:
        return [[-1, 1, pz_val], [1, 1, pz_val]]
    else:
        sym_cfg = c.get("symmetry", {})
        px = int(sym_cfg.get("px", 0))
        py = int(sym_cfg.get("py", 0))
        return [[px, py, pz_val]]


def calc_df_refine(f: float, q_coarse: float) -> float:
    q_safe = max(float(q_coarse), 1.0)
    return float(np.clip(10.0 * float(f) / q_safe, 0.01, 0.06))


def calc_df_mode(f: float, q_refine: float) -> float:
    q_safe = max(float(q_refine), 1.0)
    return float(np.clip(2.0 * float(f) / q_safe, 0.005, 0.025))


def calc_tmax_fine(f: float, q_est: float, q_expected: float = 1e4) -> float:
    f_safe = max(float(f), 0.1)
    q_safe = max(float(q_est), 20.0)
    t_q = min(2000.0, max(450.0, 5.0 * q_safe / f_safe))
    if q_expected > 0:
        t_exp = min(2000.0, max(450.0, 5.0 * float(q_expected) / f_safe))
        return float(round(min(t_q, t_exp), 1))
    return float(round(t_q, 1))


def derive_params(user_cfg: Dict[str, Any] | None = None, verbose: bool = True) -> Dict[str, Any]:
    """
    spv_gpu/config.py および spv/run_sim.py と条件を揃えた全計算パラメータを導出する。
    """
    c = normalize_cfg(user_cfg)
    raw = user_cfg or {}
    stype = int(c["structure_type"])
    st = c["structure"]
    n = float(st["n"])
    period = float(st["period"])

    f_min, f_max = float(c["f_range"][0]), float(c["f_range"][1])
    if f_min <= 0 or f_max <= f_min:
        raise ValueError(f"f_range が不正です: {c['f_range']}")

    d_min = min_feature(c)
    res_floor = math.ceil(12.0 * n * f_max)

    quality = str(c.get("quality", "standard")).lower()
    if "accuracy" in raw and isinstance(raw["accuracy"], dict):
        acc_coarse = float(raw["accuracy"].get("coarse", 2.0))
        acc_fine = float(raw["accuracy"].get("fine", 4.0))
    else:
        acc_coarse, acc_fine = (
            (1.5, 3.0) if quality == "fast" else ((2.5, 4.5) if quality == "high" else (2.0, 4.0))
        )

    if stype == 6:
        res_harminv = max(int(round(27.5 * acc_coarse)), res_floor)
        res_mode = max(int(round(20.0 * acc_fine)), res_floor)
    elif stype in (3, 4, 5):
        res_harminv = max(int(round(30.0 * acc_coarse)), res_floor)
        res_mode = max(int(round(22.5 * acc_fine)), res_floor)
    else:
        res_harminv = max(int(round(25.0 * acc_coarse)), res_floor)
        res_mode = max(int(round(20.0 * acc_fine)), res_floor)

    res_harminv = min(res_harminv, 90)
    res_mode = min(res_mode, 100)

    if raw.get("res_harminv") is not None:
        res_harminv = int(raw["res_harminv"])
    if raw.get("res_mode") is not None:
        res_mode = int(raw["res_mode"])

    fcen = 0.5 * (f_min + f_max)
    df = 1.2 * (f_max - f_min)
    cutoff = CUTOFF

    if stype == 6:
        dist_xy, dist_z, pml_w_base = 0.35, 0.55, 0.35
    elif stype in (3, 4, 5):
        dist_xy, dist_z, pml_w_base = 0.70, 0.60, 0.10
    else:
        dist_xy, dist_z, pml_w_base = 0.50, 0.60, 0.35

    dist_to_pml = [dist_xy, dist_xy, dist_z]
    pml_w = [pml_w_base, pml_w_base, pml_w_base]
    lx_s, ly_s, lz_s = _compute_structure_bounding_box(c)

    def _calc_sp(res_um: int) -> Tuple[List[float], List[float]]:
        if bool(c.get("legacy_spv_cell", False)):
            if stype in (6, 0):
                return [lx_s + 2.0 * dist_xy, ly_s + 2.0 * dist_xy, lz_s + 2.0 * dist_z], pml_w
            return [lx_s + dist_xy, ly_s + dist_xy, lz_s + dist_z], pml_w

        # spv_gpu/config.py と同一の空気余白 + PML厚 + グリッド整合セルサイズ
        pml_cells = max(10, int(math.ceil(pml_w_base * res_um)))
        pml_eff = pml_cells / float(res_um)
        if stype in (6, 0):
            sx = lx_s + 2.0 * (dist_xy + pml_eff)
            sy = ly_s + 2.0 * (dist_xy + pml_eff)
            sz = lz_s + 2.0 * (dist_z + pml_eff)
        else:
            sx = lx_s + dist_xy + 2.0 * pml_eff
            sy = ly_s + dist_xy + 2.0 * pml_eff
            sz = lz_s + dist_z + 2.0 * pml_eff
        nx = max(16, int(round(sx * res_um)))
        ny = max(16, int(round(sy * res_um)))
        nz_full = max(16, int(round(sz * res_um)))
        nx += nx % 2
        ny += ny % 2
        nz_full += nz_full % 2
        return [nx / float(res_um), ny / float(res_um), nz_full / float(res_um)], [pml_eff, pml_eff, pml_eff]

    sp_harminv, pml_w_harminv = _calc_sp(res_harminv)
    sp_mode, pml_w_mode = _calc_sp(res_mode)

    # パリティ展開 (type 3/4 は z 非対称のため ['o'] に統一)
    if stype in (3, 4):
        parities = ["o"]
        if verbose and list(c.get("parities", [])) != ["o"]:
            print("[情報] structure_type=3,4 は z 非対称構造のため parities=['o'] (pz=0) に自動統一します。")
    else:
        raw_parities = c.get("parities")
        if raw_parities is not None:
            parities = [p if isinstance(p, str) else PARITY_MAP[int(p)] for p in raw_parities]
        else:
            pz_cfg = c.get("symmetry", {}).get("pz", "auto")
            if pz_cfg == "auto":
                parities = ["+", "-"]
            elif isinstance(pz_cfg, (list, tuple)):
                parities = [p if isinstance(p, str) else PARITY_MAP[int(p)] for p in pz_cfg]
            else:
                parities = [pz_cfg if isinstance(pz_cfg, str) else PARITY_MAP[int(pz_cfg)]]

    default_src_pos = [[0.0, 0.0, 0.0], [0.05, 0.05, 0.0]]
    src_pos_harminv = np.array(raw.get("src_pos", default_src_pos), dtype=float)
    m_pos_harminv = raw.get("m_pos", get_default_monitor_positions(c))

    q_exp = float(c.get("Q_expected", 1e4))
    tmax_coarse = float(raw.get("tmax_coarse", min(2000.0, max(300.0, q_exp / max(fcen, 0.1)))))

    return {
        "structure_type": stype,
        "structure": st,
        "n": n,
        "period": period,
        "d_min": d_min,
        "feat_min": d_min,
        "f_min": f_min,
        "f_max": f_max,
        "fcen": fcen,
        "df": df,
        "cutoff": cutoff,
        "res_harminv": int(res_harminv),
        "res_mode": int(res_mode),
        "dist_to_pml": dist_to_pml,
        "pml_w": pml_w_harminv,
        "pml_w_harminv": pml_w_harminv,
        "pml_w_mode": pml_w_mode,
        "sp": sp_harminv,
        "sp_harminv": sp_harminv,
        "sp_mode": sp_mode,
        "parities": parities,
        "n_modes": int(c.get("n_modes", 2)),
        "run_mode": str(c.get("run_mode", "ALL")).upper(),
        "src_pos": src_pos_harminv,
        "m_pos": m_pos_harminv,
        "Q_expected": q_exp,
        "tmax_coarse": tmax_coarse,
        "use_decay_stop": bool(c.get("use_decay_stop", True)),
        "decay_dt": float(c.get("decay_dt", 50.0)),
        "decay_by": float(c.get("decay_by", 1e-5)),
        "Ath": ATH,
        "Q_min": Q_MIN,
        "refine_tol_fac": REFINE_MATCH_FACTOR,
        "merge_refine": bool(c.get("merge_refine_mode", True)),
        "n_shots": int(c.get("n_shots", 1)),
        "seed": int(c.get("random_seed", 42)),
        "save_h5": bool(c.get("save_h5", False)),
        "out_dir": str(c["out_dir"]),
    }


def derive_parameters(user_cfg: Dict[str, Any] | None = None, verbose: bool = True) -> Dict[str, Any]:
    """spv_gpu.config.derive_parameters 互換エイリアス。"""
    return derive_params(user_cfg=user_cfg, verbose=verbose)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_runtime_cfg(user_cfg: Dict[str, Any], path: str | None = None) -> str:
    """MPI サブプロセスが読み込めるよう設定を JSON 保存する。"""
    if path is None:
        out_dir = str(normalize_cfg(user_cfg)["out_dir"])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "spv_runtime_cfg.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(user_cfg, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def load_runtime_cfg(default_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """SPV_USER_CFG_JSON が指定されていれば読み込み、なければ default_cfg を返す。"""
    cfg_path = os.environ.get("SPV_USER_CFG_JSON", "").strip()
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if "f_range" in loaded and isinstance(loaded["f_range"], list):
            loaded["f_range"] = tuple(loaded["f_range"])
        return loaded
    return copy.deepcopy(default_cfg if default_cfg is not None else cfg)
