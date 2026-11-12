"""
STEP 2 automatic mode clustering, ranking, and output file generation (Spec A §5.7 & Spec B1 §3).
Replaces manual inspection of `best condition {date}.txt` while preserving 100% file format compatibility.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_TSV_COLUMNS = [
    "freq",
    "Q",
    "Re[A]",
    "Im[A]",
    "|A|",
    "Re[Er]",
    "excon",
    "detcon",
    "sym",
    "file name",
]


def cluster_and_select_modes(
    harminv_df: pd.DataFrame,
    n_modes_select: int = 3,
    min_detections: int = 2,
    rel_freq_tol_floor: float = 2e-3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    全モニタ点・全成分・各パリティの Harminv 検出結果 `harminv_df` から
    近接周波数 `|Δf| < max(f / (2*Q), f * rel_freq_tol_floor)` でモードをクラスタリングし、
    Q中央値と最大振幅で上位 `n_modes_select` 個の共振モードを自動選出する。

    Returns
    -------
    selected_df : pd.DataFrame
        上位 K モードの代表行 DataFrame (`best condition {date}.txt` 互換カラム + 統計量カラム付き)
    all_clusters_df : pd.DataFrame
        全クラスタの統計一覧 DataFrame
    """
    if harminv_df is None or harminv_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = harminv_df.copy()
    if "pz" not in df.columns:
        df["pz"] = 0

    cluster_records: List[Dict[str, Any]] = []

    # パリティ (pz) 毎に独立してクラスタリング
    for pz_val, grp in df.groupby("pz"):
        grp_sorted = grp.sort_values(by="freq", ascending=True).reset_index(drop=True)
        used = np.zeros(len(grp_sorted), dtype=bool)

        for i in range(len(grp_sorted)):
            if used[i]:
                continue
            f_seed = float(grp_sorted.loc[i, "freq"])
            q_seed = max(float(grp_sorted.loc[i, "Q"]), 20.0)
            tol = max(f_seed / (2.0 * q_seed), f_seed * rel_freq_tol_floor)

            diffs = np.abs(grp_sorted["freq"].to_numpy(dtype=np.float64) - f_seed)
            members_mask = (~used) & (diffs <= tol)
            used[members_mask] = True
            members = grp_sorted[members_mask]

            # 代表行: クラスタ内で |A| * sqrt(Q) が最大の検出行
            scores = members["|A|"].to_numpy(dtype=np.float64) * np.sqrt(
                np.maximum(members["Q"].to_numpy(dtype=np.float64), 1.0)
            )
            best_local_idx = int(np.argmax(scores))
            rep_row = members.iloc[best_local_idx]

            f_vals = members["freq"].to_numpy(dtype=np.float64)
            q_vals = members["Q"].to_numpy(dtype=np.float64)
            a_vals = members["|A|"].to_numpy(dtype=np.float64)

            q75, q25 = np.percentile(q_vals, [75, 25]) if len(q_vals) > 1 else (q_vals[0], q_vals[0])

            rec = {
                "freq": float(np.median(f_vals)),
                "f_std": float(np.std(f_vals)),
                "Q": float(np.median(q_vals)),
                "Q_max": float(np.max(q_vals)),
                "Q_iqr": float(q75 - q25),
                "Re[A]": float(rep_row.get("Re[A]", rep_row["|A|"])),
                "Im[A]": float(rep_row.get("Im[A]", 0.0)),
                "|A|": float(np.max(a_vals)),
                "Re[Er]": float(np.min(members["Re[Er]"].to_numpy(dtype=np.float64))),
                "excon": str(rep_row.get("excon", "multi_src")),
                "detcon": str(rep_row.get("detcon", "pt0")),
                "sym": str(rep_row.get("sym", "none")),
                "file name": str(rep_row.get("file name", "harminv.txt")),
                "pz": int(pz_val),
                "n_detections": int(len(members)),
                "df_refine": float(max(5.0 * f_seed / max(float(np.median(q_vals)), 50.0), f_seed / 200.0)),
                "df_mode": float(max(2.0 * f_seed / max(float(np.median(q_vals)), 50.0), f_seed / 500.0)),
            }
            cluster_records.append(rec)

    if not cluster_records:
        return pd.DataFrame(), pd.DataFrame()

    clusters_df = pd.DataFrame(cluster_records)

    # 複数点・複数成分で安定して検出されたクラスタを優先（該当がなければ全クラスタから選出）
    multi_det = clusters_df[clusters_df["n_detections"] >= min_detections]
    pool_df = multi_det if len(multi_det) >= 1 else clusters_df

    # ランキングスコア: Q中央値 × log10(1 + |A| / max|A| * 10)
    max_a_global = max(float(pool_df["|A|"].max()), 1e-30)
    pool_df = pool_df.copy()
    pool_df["rank_score"] = pool_df["Q"] * np.log10(1.0 + 10.0 * (pool_df["|A|"] / max_a_global))
    pool_df = pool_df.sort_values(by=["rank_score", "Q", "|A|"], ascending=[False, False, False]).reset_index(
        drop=True
    )

    selected_df = pool_df.head(n_modes_select).reset_index(drop=True)
    clusters_df = clusters_df.sort_values(by=["Q", "|A|"], ascending=[False, False]).reset_index(drop=True)
    return selected_df, clusters_df


def save_best_condition_files(
    selected_df: pd.DataFrame,
    out_dir: str,
    date_str: str,
    stage_suffix: str = "",
) -> Tuple[str, str]:
    """
    `spv/my_analysis.py` と完全互換のタブ区切り形式で `best condition {date}.txt` と
    詳細統計付き CSV を出力する。
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{stage_suffix}" if stage_suffix else ""
    best_txt_path = os.path.join(out_dir, f"best condition {date_str}{suffix}.txt")
    detail_csv_path = os.path.join(out_dir, f"mode_clusters_{date_str}{suffix}.csv")

    if selected_df is None or selected_df.empty:
        empty_tsv = pd.DataFrame(columns=LEGACY_TSV_COLUMNS)
        empty_tsv.to_csv(best_txt_path, sep="\t", index=True)
        return best_txt_path, detail_csv_path

    # レガシー互換 TSV 出力
    tsv_df = selected_df[LEGACY_TSV_COLUMNS].copy()
    tsv_df.to_csv(best_txt_path, sep="\t", index=True)

    # 統計量を含む完全版 CSV 出力
    selected_df.to_csv(detail_csv_path, index=False)

    # ステージサフィックスなし（STEP 3 が直接参照する標準ファイル名）も同時更新
    if stage_suffix:
        canonical_path = os.path.join(out_dir, f"best condition {date_str}.txt")
        tsv_df.to_csv(canonical_path, sep="\t", index=True)

    return best_txt_path, detail_csv_path


def plot_harminv_summary(
    harminv_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    fmin: float,
    fmax: float,
    out_png_path: str,
    title_prefix: str = "",
) -> None:
    """
    Harminv で検出された全共振ピークの (周波数 vs Q値) 散布図および周波数ヒストグラムを描画・保存する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)

    ax_q, ax_hist = axes
    if harminv_df is not None and not harminv_df.empty:
        for sym_label, grp in harminv_df.groupby("sym"):
            ax_q.scatter(
                grp["freq"],
                grp["Q"],
                s=np.clip(np.log10(grp["|A|"] / max(grp["|A|"].max(), 1e-30) * 1e4 + 1.1) * 25.0, 15.0, 120.0),
                alpha=0.65,
                label=f"All detections ({sym_label})",
            )
        ax_hist.hist(harminv_df["freq"], bins=30, range=(fmin, fmax), color="steelblue", alpha=0.75, edgecolor="black")

    if selected_df is not None and not selected_df.empty:
        ax_q.scatter(
            selected_df["freq"],
            selected_df["Q"],
            s=160,
            marker="*",
            color="crimson",
            edgecolor="black",
            zorder=5,
            label="Selected modes (Top K)",
        )
        for idx, row in selected_df.iterrows():
            ax_q.annotate(
                f"M{idx}: f={row['freq']:.4f}\nQ={row['Q']:.0f}",
                (row["freq"], row["Q"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                fontweight="bold",
                color="darkred",
            )

    ax_q.set_yscale("log")
    ax_q.set_xlim(fmin, fmax)
    ax_q.set_xlabel("Frequency [Meep units: c / um]")
    ax_q.set_ylabel("Quality factor Q")
    ax_q.set_title(f"{title_prefix} Frequency vs Q".strip())
    ax_q.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_q.legend(loc="best", fontsize=8)

    ax_hist.set_xlim(fmin, fmax)
    ax_hist.set_xlabel("Frequency [Meep units: c / um]")
    ax_hist.set_ylabel("Detection count")
    ax_hist.set_title(f"{title_prefix} Detection Histogram".strip())
    ax_hist.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
