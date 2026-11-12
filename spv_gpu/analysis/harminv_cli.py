"""
Wrapper for upstream Harminv CLI (`harminv -t dt fmin-fmax`) with parallel execution
and compatibility with `spv/my_analysis.py` (Spec B1 §8).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import math
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


HARMINV_COLUMNS = ["freq", "decay", "Q", "|A|", "phase", "error"]


def check_harminv_cli() -> Tuple[bool, str]:
    """`harminv` コマンドラインバイナリが PATH 上に存在するか確認し、バージョン文字列を返す。"""
    exe = shutil.which("harminv")
    if exe is None:
        return False, "harminv binary not found in PATH"
    for flag in ("-V", "--version", "-h"):
        try:
            res = subprocess.run(
                [exe, flag],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            combined = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
            if combined:
                first_line = combined.splitlines()[0].strip()
                if "invalid option" not in first_line.lower() and "unrecognized option" not in first_line.lower():
                    return True, first_line
        except Exception:
            continue
    return True, f"harminv ({exe})"


def _fallback_matrix_pencil_harminv(
    t_series: np.ndarray,
    dt: float,
    fmin: float,
    fmax: float,
    max_poles: int = 24,
) -> pd.DataFrame:
    """
    `harminv` CLI バイナリが未インストールの環境（ローカル検証用等）向けの
    SVD Matrix Pencil / Prony 法フォールバック実装。
    Colab 環境では `apt-get install -y harminv` により本家 CLI が優先使用される。
    """
    y = np.asarray(t_series, dtype=np.float64)
    n = len(y)
    if n < 32 or np.max(np.abs(y)) < 1e-30:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    # 計算量抑制のため長すぎる場合は適度に間引く
    dec = max(1, int(math.floor(1.0 / (10.0 * dt * max(fmax, 0.1)))))
    if n // dec > 2000:
        dec = max(dec, n // 2000)
    if dec > 1:
        y = y[::dec]
        dt = dt * dec
        n = len(y)

    # 帯域内の主要な極をハンケル行列 SVD で抽出
    l_pencil = min(n // 3, 120)
    if l_pencil < 8:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    y_mat = np.lib.stride_tricks.sliding_window_view(y, l_pencil + 1)
    y1 = y_mat[:, :-1]
    y2 = y_mat[:, 1:]

    u, s, vh = np.linalg.svd(y1, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    # 有効ランクの決定
    rank = int(np.sum(s > s[0] * 1e-4))
    rank = max(2, min(rank, max_poles * 2, len(s)))

    u_r = u[:, :rank]
    s_r = s[:rank]
    v_r = vh[:rank, :].T

    z_mat = (1.0 / s_r)[:, None] * (u_r.T @ y2 @ v_r)
    eigvals = np.linalg.eigvals(z_mat)

    rows = []
    t_vec = np.arange(n, dtype=np.float64) * dt
    for z in eigvals:
        if np.abs(z) <= 1e-12:
            continue
        s_pole = np.log(z) / dt
        freq = float(np.imag(s_pole) / (2.0 * math.pi))
        decay = float(-np.real(s_pole) / (2.0 * math.pi))
        if fmin <= freq <= fmax and decay > -1e-6:
            q_val = freq / (2.0 * max(abs(decay), 1e-12))
            # 最小二乗で複素振幅を推定
            basis = np.exp(s_pole * t_vec)
            denom = np.vdot(basis, basis).real
            amp_c = np.vdot(basis, y) / max(denom, 1e-30) * 2.0
            amp = float(np.abs(amp_c))
            phase = float(np.angle(amp_c) / (2.0 * math.pi))
            err = float(1.0 / max(rank * 10.0, 100.0))
            rows.append([freq, decay, q_val, amp, phase, err])

    if not rows:
        return pd.DataFrame(columns=HARMINV_COLUMNS)
    return pd.DataFrame(rows, columns=HARMINV_COLUMNS)


def run_harminv(
    t_series: np.ndarray,
    dt: float,
    fmin: float,
    fmax: float,
    q_min: float = 20.0,
    err_max: float = 0.1,
    ath: float = 1e-3,
    max_samples: int = 200_000,
) -> pd.DataFrame:
    """
    仕様書B1 §8 に従い、1本の時系列データ `t_series` (t > t_src_end 切り出し済み) に対して
    本家 `harminv -t {dt} {fmin}-{fmax}` CLI を実行し、フィルタ済み DataFrame を返す。
    """
    arr = np.asarray(t_series, dtype=np.float64).ravel()
    arr = arr - np.mean(arr)
    if len(arr) < 16 or np.max(np.abs(arr)) < 1e-30:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    # 仕様書B1 §8: サンプル数 N > 2e5 の場合はナイキスト条件 1/(dt_dec * fmax) >= 8 を保ちつつ間引く
    if len(arr) > max_samples:
        max_dec = max(1, int(math.floor(1.0 / (8.0 * dt * max(fmax, 1e-6)))))
        dec = min(max_dec, int(math.ceil(len(arr) / float(max_samples))))
        if dec > 1:
            arr = arr[::dec]
            dt = dt * dec

    has_cli, _ = check_harminv_cli()
    if has_cli:
        cmd = ["harminv", "-t", f"{dt:.10g}", f"{fmin:.8g}-{fmax:.8g}"]
        input_text = "\n".join(f"{v:.10e}" for v in arr) + "\n"
        res = subprocess.run(cmd, input=input_text, text=True, capture_output=True)
        if res.returncode == 0 and res.stdout.strip():
            lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
            if len(lines) > 1:
                df = pd.read_csv(
                    io.StringIO("\n".join(lines)),
                     sep=",",
                    skipinitialspace=True,
                    names=HARMINV_COLUMNS,
                    header=0,
                )
            else:
                df = pd.DataFrame(columns=HARMINV_COLUMNS)
        else:
            df = pd.DataFrame(columns=HARMINV_COLUMNS)
    else:
        df = _fallback_matrix_pencil_harminv(arr, dt, fmin, fmax)

    if df.empty:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    for col in HARMINV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["freq", "Q", "|A|", "error"]).copy()

    if df.empty:
        return pd.DataFrame(columns=HARMINV_COLUMNS)

    # 正の周波数・誤差閾値・Q値閾値・相対振幅閾値によるポストフィルタ (仕様書B1 §8)
    df["freq"] = df["freq"].abs()
    df["Q"] = df["Q"].abs()
    df = df[(df["freq"] >= fmin) & (df["freq"] <= fmax)]
    df = df[(df["error"] < err_max) & (df["Q"] > q_min)]
    if not df.empty:
        max_amp = float(df["|A|"].max())
        df = df[df["|A|"] >= ath * max_amp]

    return df.reset_index(drop=True)


def analyze_simulation_with_harminv(
    time_array: np.ndarray,
    monitor_traces: np.ndarray,
    monitor_positions: List[Tuple[float, float, float]],
    monitor_components: List[str],
    dt_rec: float,
    t_src_end: float,
    fmin: float,
    fmax: float,
    pz: int = 0,
    q_min: float = 20.0,
    err_max: float = 0.1,
    ath: float = 1e-3,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    FDTDXSimulationResult の全モニタ点 × 全記録成分の時系列に対して
    `t > t_src_end` を切り出し、並列で `run_harminv` を実行して統合 DataFrame を返す。
    `spv/my_analysis.py` 互換カラム (`freq`, `Q`, `Re[A]`, `Im[A]`, `|A|`, `Re[Er]`,
    `excon`, `detcon`, `sym`, `file name`) をすべて付与する。
    """
    mask = time_array > t_src_end
    if not np.any(mask):
        # 万一 t_src_end より前に打ち切られた場合は後半 50% を使用
        half_idx = len(time_array) // 2
        sliced_traces = monitor_traces[half_idx:]
    else:
        sliced_traces = monitor_traces[mask]

    n_samples, n_pts, n_comps = sliced_traces.shape
    tasks: List[Tuple[int, int, np.ndarray]] = []
    for p_idx in range(n_pts):
        for c_idx in range(n_comps):
            tasks.append((p_idx, c_idx, sliced_traces[:, p_idx, c_idx]))

    def _worker(task_tuple: Tuple[int, int, np.ndarray]) -> pd.DataFrame:
        p_i, c_i, series = task_tuple
        sub_df = run_harminv(
            t_series=series,
            dt=dt_rec,
            fmin=fmin,
            fmax=fmax,
            q_min=q_min,
            err_max=err_max,
            ath=ath,
        )
        if sub_df.empty:
            return sub_df

        comp_name = monitor_components[c_i]
        pos = monitor_positions[p_i]
        sym_str = "even" if pz == 1 else ("odd" if pz == -1 else "none")

        # spv/my_analysis.py 互換カラムの構築
        phase_rad = 2.0 * math.pi * sub_df["phase"].to_numpy(dtype=np.float64)
        amp_arr = sub_df["|A|"].to_numpy(dtype=np.float64)
        sub_df["Re[A]"] = amp_arr * np.cos(phase_rad)
        sub_df["Im[A]"] = amp_arr * np.sin(phase_rad)
        sub_df["Re[Er]"] = sub_df["error"]
        sub_df["excon"] = f"multi_src_{sym_str}"
        sub_df["detcon"] = f"pt{p_i}_{comp_name}_({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
        sub_df["sym"] = sym_str
        sub_df["pz"] = pz
        sub_df["mon_idx"] = p_i
        sub_df["comp"] = comp_name
        sub_df["file name"] = f"harminv_{sym_str}_pt{p_i}_{comp_name}.txt"
        return sub_df

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        dfs = list(pool.map(_worker, tasks))

    valid_dfs = [d for d in dfs if not d.empty]
    if not valid_dfs:
        cols = HARMINV_COLUMNS + [
            "Re[A]",
            "Im[A]",
            "Re[Er]",
            "excon",
            "detcon",
            "sym",
            "pz",
            "mon_idx",
            "comp",
            "file name",
        ]
        return pd.DataFrame(columns=cols)

    merged = pd.concat(valid_dfs, ignore_index=True)
    return merged.sort_values(by=["Q", "|A|"], ascending=[False, False]).reset_index(drop=True)


def write_meep_compatible_harminv_log(df: pd.DataFrame, out_txt_path: str) -> None:
    """
    既存の `spv/my_analysis.py` (`read_harminv_file`) でそのまま読み込める
    `harminv0:` 形式のログファイルを出力する。
    """
    lines = [
        "harminv0: frequency, imag. freq., Q, |amp|, amplitude, error",
    ]
    for _, row in df.iterrows():
        f_val = float(row["freq"])
        d_val = float(row.get("decay", -f_val / (2.0 * max(float(row["Q"]), 1.0))))
        q_val = float(row["Q"])
        a_val = float(row["|A|"])
        re_a = float(row.get("Re[A]", a_val))
        im_a = float(row.get("Im[A]", 0.0))
        err_val = float(row.get("error", row.get("Re[Er]", 1e-5)))
        sign_a = "+" if im_a >= 0 else ""
        lines.append(
            f"harminv0: {f_val:.8f}, {d_val:.6e}, {q_val:.4f}, {a_val:.6e}, "
            f"{re_a:.6e}{sign_a}{im_a:.6e}i, {err_val:.6e}+0.0e+00i"
        )
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
