"""
Benchmarking, draft structure checks, and CPU-vs-GPU discrepancy comparison tools
for spv_colab_sync. Stored in the repository so notebooks stay minimal.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg_mod
from .pipeline import detect_host_resources, run_mpi_pipeline


def run_scaling_benchmark(
    max_np: Optional[int] = None,
    base_out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    NP = 1, 2, 4, ..., max_np で STEP 1 粗探索を実行し、並列効率を計測する。
    """
    if max_np is None:
        env_np = os.environ.get("NP")
        max_np = int(env_np) if env_np else detect_host_resources(verbose=False)[0]

    np_list: List[int] = []
    val = 1
    while val < max_np:
        np_list.append(val)
        val *= 2
    if max_np not in np_list:
        np_list.append(max_np)

    out_root = base_out_dir or os.environ.get("SPV_OUT_DIR", "./spv_colab_sync_results")
    bench_dir = os.path.join(os.path.abspath(out_root), "scaling_benchmark")
    os.makedirs(bench_dir, exist_ok=True)

    records = []
    for np_val in np_list:
        case_dir = os.path.join(bench_dir, f"np_{np_val}")
        os.makedirs(case_dir, exist_ok=True)
        c = cfg_mod.get_default_config()
        c["parities"] = ["+"]
        c["quality"] = "fast"
        c["accuracy"] = {"coarse": 1.5, "fine": 3.0}
        c["run_mode"] = "HARMINV"
        c["tmax_coarse"] = 300
        c["output_dir"] = case_dir

        print(f"=== スケーリング計測中: NP={np_val} ===")
        t0 = time.time()
        run_mpi_pipeline(c, np_procs=np_val, stream_output=False)
        elapsed = time.time() - t0
        records.append({"NP": np_val, "elapsed_sec": round(elapsed, 2)})
        print(f"  -> NP={np_val}: {elapsed:.2f} sec")

    df_scale = pd.DataFrame(records)
    t1 = float(df_scale.loc[0, "elapsed_sec"])
    df_scale["speedup"] = (t1 / df_scale["elapsed_sec"]).round(2)
    df_scale["efficiency_pct"] = ((df_scale["speedup"] / df_scale["NP"]) * 100.0).round(1)

    scale_csv = os.path.join(bench_dir, "scaling_results.csv")
    df_scale.to_csv(scale_csv, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_scale["NP"], df_scale["speedup"], "o-", label="Measured Speedup")
    ax.plot(df_scale["NP"], df_scale["NP"], "--", color="gray", label="Ideal Linear")
    ax.set_xlabel("MPI Processes (NP)")
    ax.set_ylabel("Speedup")
    ax.set_title("Meep MPI Scaling (spv_colab_sync)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(bench_dir, "scaling_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[並列効率計測結果]")
    print(df_scale.to_string(index=False))
    return df_scale


def run_all_structures_draft_test(
    np_procs: Optional[int] = None,
    base_out_dir: Optional[str] = None,
    stypes: Tuple[int, ...] = (3, 4, 5, 6),
) -> pd.DataFrame:
    """
    全構造 (type 3, 4, 5, 6) を fast 設定で一括実行し、モード抽出を確認する。
    """
    out_root = base_out_dir or os.environ.get("SPV_OUT_DIR", "./spv_colab_sync_results")
    base_out = os.path.join(os.path.abspath(out_root), "draft_all_structures")
    os.makedirs(base_out, exist_ok=True)

    f_ranges = {
        0: (0.55, 0.75),
        3: (2.15, 2.35),
        4: (2.15, 2.35),
        5: (2.15, 2.35),
        6: (0.55, 0.75),
    }

    all_summaries = []
    for stype in stypes:
        st_out = os.path.join(base_out, f"type_{stype}")
        os.makedirs(st_out, exist_ok=True)
        c = cfg_mod.get_default_config()
        c["structure_type"] = stype
        c["structure"] = {}
        c["f_range"] = f_ranges[stype]
        c["parities"] = ["o"] if stype in (3, 4) else ["+", "-"]
        c["quality"] = "fast"
        c["accuracy"] = {"coarse": 1.5, "fine": 3.0}
        c["n_modes"] = 1
        c["run_mode"] = "ALL"
        c["output_dir"] = st_out

        print(f"\n================ 検証実行: structure_type = {stype} ================")
        res = run_mpi_pipeline(c, np_procs=np_procs, stream_output=True)
        df_s = res["summary_df"]
        if not df_s.empty:
            df_s = df_s.copy()
            df_s.insert(0, "structure_type", stype)
            all_summaries.append(df_s)

    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        print("\n=== 全構造 Draft 実行結果サマリー ===")
        print(final_df.to_string(index=False))
        return final_df
    return pd.DataFrame()


def _extract_pz(row: pd.Series) -> int:
    if "pz" in row and pd.notna(row["pz"]):
        return int(row["pz"])
    pz_map = {"+": 1, "-": -1, "o": 0}
    if "parity" in row and pd.notna(row["parity"]):
        return int(pz_map.get(str(row["parity"]).strip(), 0))
    if "sym" in row and pd.notna(row["sym"]):
        return int(pz_map.get(str(row["sym"]).strip()[-1], 0))
    return 0


def compare_cpu_vs_gpu(
    cpu_summary_csv: str,
    gpu_summary_csv: str,
    out_dir: Optional[str] = None,
    max_rel_freq_diff: float = 0.05,
) -> pd.DataFrame:
    """
    spv_colab_sync (CPU Meep) と spv_gpu (GPU FDTDX) の summary.csv を読み込み、
    同一パリティ (pz) ごとに最も周波数が近いモード同士を突合して乖離 (Δf/f, ΔQ/Q) を算出する。
    """
    df_cpu = pd.read_csv(cpu_summary_csv)
    df_gpu = pd.read_csv(gpu_summary_csv)
    if df_cpu.empty or df_gpu.empty:
        print("[警告] CPU または GPU の summary.csv が空のため比較できません。")
        return pd.DataFrame()

    df_cpu = df_cpu.copy()
    df_gpu = df_gpu.copy()
    df_cpu["_pz"] = df_cpu.apply(_extract_pz, axis=1)
    df_gpu["_pz"] = df_gpu.apply(_extract_pz, axis=1)

    rows: List[Dict[str, Any]] = []
    used_gpu = set()

    for _, c_row in df_cpu.iterrows():
        pz = int(c_row["_pz"])
        f_cpu = float(c_row["freq"])
        q_cpu = float(c_row["Q"])

        cand = df_gpu[(df_gpu["_pz"] == pz) & (~df_gpu.index.isin(used_gpu))]
        if cand.empty:
            continue

        rel_diffs = (cand["freq"].astype(float) - f_cpu).abs() / max(f_cpu, 1e-12)
        best_idx = int(rel_diffs.idxmin())
        best_rel_f = float(rel_diffs.loc[best_idx])
        if best_rel_f > max_rel_freq_diff:
            continue

        used_gpu.add(best_idx)
        g_row = df_gpu.loc[best_idx]
        f_gpu = float(g_row["freq"])
        q_gpu = float(g_row["Q"])
        rel_q_err = abs(q_gpu - q_cpu) / max(q_cpu, 1e-12)

        rows.append({
            "pz": pz,
            "freq_cpu": f_cpu,
            "freq_gpu": f_gpu,
            "abs_df": abs(f_gpu - f_cpu),
            "rel_df_pct": best_rel_f * 100.0,
            "Q_cpu": q_cpu,
            "Q_gpu": q_gpu,
            "rel_dQ_pct": rel_q_err * 100.0,
        })

    comp_df = pd.DataFrame(rows)
    if out_dir and not comp_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        comp_csv = os.path.join(out_dir, "cpu_vs_gpu_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
        ax1.scatter(comp_df["freq_cpu"], comp_df["freq_gpu"], s=70)
        lims = [min(comp_df["freq_cpu"].min(), comp_df["freq_gpu"].min()), max(comp_df["freq_cpu"].max(), comp_df["freq_gpu"].max())]
        ax1.plot(lims, lims, "--", color="gray")
        ax1.set_xlabel("CPU Meep freq")
        ax1.set_ylabel("GPU FDTDX freq")
        ax1.set_title("Resonance Frequency Comparison")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(comp_df["Q_cpu"], comp_df["Q_gpu"], s=70, color="darkorange")
        qlims = [min(comp_df["Q_cpu"].min(), comp_df["Q_gpu"].min()), max(comp_df["Q_cpu"].max(), comp_df["Q_gpu"].max())]
        ax2.plot(qlims, qlims, "--", color="gray")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("CPU Meep Q")
        ax2.set_ylabel("GPU FDTDX Q")
        ax2.set_title("Q-factor Comparison")
        ax2.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cpu_vs_gpu_comparison.png"), bbox_inches="tight")
        plt.close(fig)

    return comp_df
