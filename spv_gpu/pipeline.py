"""
End-to-end automated pipeline (STEP 1 -> STEP 2 -> STEP 2.5 -> STEP 3) for spv_gpu
with ProgressEstimator, Google Drive checkpointing, epchk visualization, mode profile
plotting, summary.csv, and run_log.txt output (Spec A §5.7-9 & Spec B1 §3-8).
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis.harminv_cli import (
    analyze_simulation_with_harminv,
    check_harminv_cli,
    write_meep_compatible_harminv_log,
)
from .analysis.select_modes import (
    cluster_and_select_modes,
    plot_harminv_summary,
    save_best_condition_files,
)
from .config import DEFAULT_BACKEND, derive_parameters
from .geometry.structures import build_structure_permittivity
from .solver.fdtdx_runner import check_gpu_environment, run_fdtdx_simulation
from .solver.symmetry import parity_label, unfold_2d_slice_z


class ProgressEstimator:
    """
    仕様書A §6 に基づく計算量ベース (`work = res^4 * T_MAX`) の進捗・残り時間推定クラス。
    """

    def __init__(self, stages: List[Tuple[str, float]]) -> None:
        self.stages = stages
        self.total_work = max(sum(w for _, w in stages), 1e-9)
        self.completed_work = 0.0
        self.t_start = time.perf_counter()
        self.stage_times: Dict[str, float] = {}

    def format_eta(self, current_stage_work_done: float = 0.0) -> str:
        done = min(self.completed_work + current_stage_work_done, self.total_work)
        frac = done / self.total_work
        elapsed = time.perf_counter() - self.t_start
        if frac > 0.01:
            eta_sec = elapsed * (1.0 - frac) / frac
            eta_str = f"{eta_sec / 60.0:.1f} min" if eta_sec >= 60 else f"{eta_sec:.0f} s"
        else:
            eta_str = "estimating..."
        return f"[{frac * 100.0:5.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta_str}]"

    def finish_stage(self, name: str, work: float, duration_sec: float) -> None:
        self.completed_work += work
        self.stage_times[name] = duration_sec


def plot_epchk_slices(
    eps_grid: np.ndarray,
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pz: int,
    out_png_path: str,
    title: str = "Permittivity Structure Check (epchk)",
) -> None:
    """
    `spv/run_sim.py` の `epchk` モードに相当する 3断面 (xy z=0, xz y=0, yz x=0) の誘電率分布画像を保存する。
    """
    xs, ys, zs = coords
    eps_3d = np.asarray(eps_grid[0], dtype=np.float32)

    iz0 = int(np.argmin(np.abs(zs - 0.0)))
    iy0 = int(np.argmin(np.abs(ys - 0.0)))
    ix0 = int(np.argmin(np.abs(xs - 0.0)))

    eps_xy = eps_3d[:, :, iz0]
    eps_xz, zs_full = unfold_2d_slice_z(eps_3d[:, iy0, :], zs, pz=pz, comp_name=None)
    eps_yz, _ = unfold_2d_slice_z(eps_3d[ix0, :, :], zs, pz=pz, comp_name=None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)

    im0 = axes[0].imshow(
        eps_xy.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="Greys",
        aspect="equal",
    )
    axes[0].set_title("xy plane (z ≈ 0)")
    axes[0].set_xlabel("x [um]")
    axes[0].set_ylabel("y [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="ε_r")

    im1 = axes[1].imshow(
        eps_xz.T,
        origin="lower",
        extent=[xs[0], xs[-1], zs_full[0], zs_full[-1]],
        cmap="Greys",
        aspect="equal",
    )
    axes[1].set_title("xz plane (y = 0)")
    axes[1].set_xlabel("x [um]")
    axes[1].set_ylabel("z [um]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="ε_r")

    im2 = axes[2].imshow(
        eps_yz.T,
        origin="lower",
        extent=[ys[0], ys[-1], zs_full[0], zs_full[-1]],
        cmap="Greys",
        aspect="equal",
    )
    axes[2].set_title("yz plane (x = 0)")
    axes[2].set_xlabel("y [um]")
    axes[2].set_ylabel("z [um]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ε_r")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def plot_mode_profile_slices(
    dft_slice_dict: Dict[str, Any],
    mode_idx: int,
    freq: float,
    q_val: float,
    sym_str: str,
    out_png_path: str,
) -> None:
    """
    STEP 3 で計算された 4断面 (`xy z=0`, `xy z=-0.06`, `xz y=0`, `yz x=0`) の
    電場エネルギー密度 D_den = 1/2 ε |E|^2 と磁場エネルギー密度 H_den = 1/2 |H|^2 を
    誘電率構造輪郭（コンター）付きでプロット・保存する。
    """
    xs = dft_slice_dict["xs"]
    ys = dft_slice_dict["ys"]
    zs = dft_slice_dict["zs_full"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), dpi=140)

    planes = [
        ("xy (z=0)", xs, ys, dft_slice_dict["Dden_xy_z0"], dft_slice_dict["Hden_xy_z0"], dft_slice_dict["eps_xy_z0"], "x [um]", "y [um]"),
        ("xy (z=±0.06)", xs, ys, dft_slice_dict["Dden_xy_zoff"], dft_slice_dict["Hden_xy_zoff"], dft_slice_dict["eps_xy_zoff"], "x [um]", "y [um]"),
        ("xz (y=0)", xs, zs, dft_slice_dict["Dden_xz_y0"], dft_slice_dict["Hden_xz_y0"], dft_slice_dict["eps_xz_y0"], "x [um]", "z [um]"),
        ("yz (x=0)", ys, zs, dft_slice_dict["Dden_yz_x0"], dft_slice_dict["Hden_yz_x0"], dft_slice_dict["eps_yz_x0"], "y [um]", "z [um]"),
    ]

    for col_i, (p_title, c1, c2, d_arr, h_arr, eps_arr, xl, yl) in enumerate(planes):
        extent = [c1[0], c1[-1], c2[0], c2[-1]]
        d_norm = d_arr / max(float(np.max(d_arr)), 1e-30)
        h_norm = h_arr / max(float(np.max(h_arr)), 1e-30)

        # 上段: 電場エネルギー密度 D_den
        ax_d = axes[0, col_i]
        im_d = ax_d.imshow(d_norm.T, origin="lower", extent=extent, cmap="inferno", aspect="equal", vmin=0.0, vmax=1.0)
        if np.ptp(eps_arr) > 0.1:
            ax_d.contour(c1, c2, eps_arr.T, levels=[0.5 * (eps_arr.min() + eps_arr.max())], colors="cyan", linewidths=0.6, alpha=0.8)
        ax_d.set_title(f"D_den: {p_title}")
        ax_d.set_xlabel(xl)
        ax_d.set_ylabel(yl)
        plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

        # 下段: 磁場エネルギー密度 H_den
        ax_h = axes[1, col_i]
        im_h = ax_h.imshow(h_norm.T, origin="lower", extent=extent, cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)
        if np.ptp(eps_arr) > 0.1:
            ax_h.contour(c1, c2, eps_arr.T, levels=[0.5 * (eps_arr.min() + eps_arr.max())], colors="white", linewidths=0.6, alpha=0.8)
        ax_h.set_title(f"H_den: {p_title}")
        ax_h.set_xlabel(xl)
        ax_h.set_ylabel(yl)
        plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Mode #{mode_idx} ({sym_str}) | f = {freq:.6f} [c/um] | Q = {q_val:.1f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def run_spv_pipeline(
    cfg: Dict[str, Any],
    backend: Optional[Dict[str, Any]] = None,
    date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    仕様書A・仕様書B1 に準拠した全自動パイプラインを実行する。
    STEP 1 (広帯域 Harminv 探索) -> STEP 2 (自動クラスタリング・モード選択) ->
    STEP 2.5 (高解像度 Harminv 精密化) -> STEP 3 (Running DFT 2D断面モード分布出力)
    """
    if backend is None:
        backend = dict(DEFAULT_BACKEND)
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    derived = derive_parameters(cfg, backend)
    parities = derived["parities"]
    run_mode = str(cfg.get("run_mode", "ALL")).upper()
    merge_refine_mode = bool(cfg.get("merge_refine_mode", False))

    # 出力先ディレクトリの構築
    from .geometry.structures import get_structure_prefix

    struct_prefix = get_structure_prefix(cfg).rstrip("/")
    base_out_dir = os.path.join(cfg.get("output_dir", "./spv_results"), struct_prefix, date_str)
    os.makedirs(base_out_dir, exist_ok=True)

    log_file_path = os.path.join(base_out_dir, "run_log.txt")
    checkpoint_path = os.path.join(base_out_dir, "checkpoint_state.json")

    def _log(msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_file_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")

    # 環境チェック
    nx_h, ny_h, nz_h = derived["grid_harminv"]
    env_info = check_gpu_environment(nx_h, ny_h, nz_h, dtype=backend.get("dtype", "float32"))
    has_harminv_bin, harminv_ver = check_harminv_cli()

    _log("=== SPV GPU Pipeline (FDTDX + Harminv CLI) Started ===")
    _log(f"Device: {env_info['platform']} ({env_info['device_kind']}) | FDTDX: {env_info['fdtdx_version']}")
    _log(f"Harminv CLI: {has_harminv_bin} ({harminv_ver})")
    _log(f"Structure: {struct_prefix}")
    _log(
        f"Derived resolutions: res_harminv={derived['res_harminv']} px/um "
        f"(grid={derived['grid_harminv']}, est_VRAM={derived['est_mem_harminv_gb']:.2f} GB), "
        f"res_mode={derived['res_mode']} px/um "
        f"(grid={derived['grid_mode']}, est_VRAM={derived['est_mem_mode_gb']:.2f} GB)"
    )
    _log(f"Parities to evaluate: {parities} | Run mode: {run_mode} | Merge STEP 2.5+3: {merge_refine_mode}")

    # ProgressEstimator のワーク定義 (work = res^4 * T_MAX)
    stage_defs: List[Tuple[str, float]] = []
    if run_mode in ("ALL", "HARMINV"):
        for pz in parities:
            w1 = (float(derived["res_harminv"]) ** 4) * float(derived["T_MAX_harminv"])
            stage_defs.append((f"STEP1_pz{pz}", w1))
    if run_mode == "ALL":
        for pz in parities:
            if merge_refine_mode:
                w25_3 = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_refine"])
                stage_defs.append((f"STEP2.5+3_pz{pz}", w25_3))
            else:
                w25 = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_refine"])
                w3 = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_mode"])
                stage_defs.append((f"STEP2.5_pz{pz}", w25))
                stage_defs.append((f"STEP3_pz{pz}", w3))
    elif run_mode == "MODE":
        for pz in parities:
            w3 = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_mode"])
            stage_defs.append((f"STEP3_pz{pz}", w3))

    estimator = ProgressEstimator(stage_defs)
    epchk_png_paths: List[str] = []
    harminv_plot_paths: List[str] = []
    mode_profile_paths: List[str] = []

    # =========================================================================
    # STEP 1 & STEP 2: 広帯域 Harminv 探索 + クラスタリング・自動モード選出
    # =========================================================================
    coarse_all_dfs: List[pd.DataFrame] = []

    if run_mode in ("ALL", "HARMINV"):
        for pz in parities:
            p_label = parity_label(pz)
            _log(f"--- [STEP 1] Broadband Harminv Search ({p_label}, res={derived['res_harminv']}) ---")
            t0_st = time.perf_counter()

            # STEP 1 では eps_averaging=False (2値セル中心判定) で高速初期化 (仕様書B1 §7)
            eps_grid_1, coords_1, _ = build_structure_permittivity(
                cfg=cfg,
                sp=derived["sp_harminv"],
                res=derived["res_harminv"],
                pz=pz,
                eps_averaging=False,
            )

            # 初回のみ epchk 構造確認画像を保存
            epchk_path = os.path.join(base_out_dir, f"epchk_{p_label}.png")
            plot_epchk_slices(eps_grid_1, coords_1, pz=pz, out_png_path=epchk_path)
            epchk_png_paths.append(epchk_path)

            comp_cfg = derived["parity_components"].get(pz, derived["parity_components"][0])
            sim_res_1 = run_fdtdx_simulation(
                eps_grid=eps_grid_1,
                coords=coords_1,
                res=derived["res_harminv"],
                fcen=derived["fcen"],
                df=derived["df"],
                t_max=derived["T_MAX_harminv"],
                src_positions=derived["src_positions"],
                src_comps=comp_cfg["src_comps"],
                src_phases=derived["src_phases"],
                mon_positions=derived["mon_positions"],
                mon_comps=comp_cfg["mon_comps"],
                pml_cells=derived["pml_cells_harminv"],
                px=derived["px"],
                py=derived["py"],
                pz=pz,
                cutoff=derived["cutoff"],
                stride=derived["stride_harminv"],
                chunk_steps=int(backend.get("chunk_steps", 2000)),
                dtype_str=backend.get("dtype", "float32"),
            )

            _log(
                f"STEP 1 FDTD finished ({p_label}): steps={sim_res_1.total_steps_run}, "
                f"early_stop={sim_res_1.stopped_early_by_decay}, "
                f"speed={sim_res_1.mcells_per_sec:.1f} Mcells*step/s, "
                f"wall={sim_res_1.wall_time_sec:.2f}s (JIT={sim_res_1.jit_compile_sec:.2f}s)"
            )

            df_harm_1 = analyze_simulation_with_harminv(
                time_array=sim_res_1.time_array,
                monitor_traces=sim_res_1.monitor_traces,
                monitor_positions=sim_res_1.monitor_positions,
                monitor_components=sim_res_1.monitor_components,
                dt_rec=sim_res_1.dt_rec,
                t_src_end=sim_res_1.t_src_end,
                fmin=derived["fmin"],
                fmax=derived["fmax"],
                pz=pz,
            )
            _log(f"STEP 1 Harminv detections ({p_label}): {len(df_harm_1)} raw peaks found.")
            if not df_harm_1.empty:
                coarse_all_dfs.append(df_harm_1)
                write_meep_compatible_harminv_log(
                    df_harm_1, os.path.join(base_out_dir, f"harminv_coarse_{p_label}.txt")
                )

            w_done = (float(derived["res_harminv"]) ** 4) * float(derived["T_MAX_harminv"])
            estimator.finish_stage(f"STEP1_pz{pz}", w_done, time.perf_counter() - t0_st)
            _log(f"Progress {estimator.format_eta()}")

        coarse_merged_df = (
            pd.concat(coarse_all_dfs, ignore_index=True) if coarse_all_dfs else pd.DataFrame()
        )
        selected_coarse_df, clusters_coarse_df = cluster_and_select_modes(
            coarse_merged_df,
            n_modes_select=int(cfg.get("n_modes_select", 3)),
        )
        best_coarse_txt, _ = save_best_condition_files(
            selected_coarse_df, out_dir=base_out_dir, date_str=date_str, stage_suffix="coarse"
        )
        harm_coarse_png = os.path.join(base_out_dir, "harminv_summary_coarse.png")
        plot_harminv_summary(
            coarse_merged_df,
            selected_coarse_df,
            fmin=derived["fmin"],
            fmax=derived["fmax"],
            out_png_path=harm_coarse_png,
            title_prefix="[STEP 1-2 Coarse]",
        )
        harminv_plot_paths.append(harm_coarse_png)
        _log(f"STEP 2 selected {len(selected_coarse_df)} candidate modes -> saved to {best_coarse_txt}")
    else:
        # run_mode == 'MODE': 既存の best condition ファイルを読み込む
        canonical_txt = os.path.join(base_out_dir, f"best condition {date_str}.txt")
        if os.path.exists(canonical_txt):
            selected_coarse_df = pd.read_csv(canonical_txt, sep="\t", index_col=0)
            if "pz" not in selected_coarse_df.columns:
                selected_coarse_df["pz"] = selected_coarse_df["sym"].map(
                    {"even": 1, "odd": -1, "none": 0}
                ).fillna(0).astype(int)
        else:
            # 候補がない場合は中心周波数を1モードとして設定
            selected_coarse_df = pd.DataFrame(
                [
                    {
                        "freq": derived["fcen"],
                        "Q": 1000.0,
                        "Re[A]": 1.0,
                        "Im[A]": 0.0,
                        "|A|": 1.0,
                        "Re[Er]": 1e-5,
                        "excon": "manual",
                        "detcon": "pt0",
                        "sym": "even" if parities[0] == 1 else "none",
                        "file name": "manual.txt",
                        "pz": parities[0],
                    }
                ]
            )
        coarse_merged_df = pd.DataFrame()

    # チェックポイント保存
    with open(checkpoint_path, "w", encoding="utf-8") as cf:
        json.dump(
            {
                "last_completed_step": "STEP2",
                "date_str": date_str,
                "n_selected_modes": len(selected_coarse_df),
            },
            cf,
            indent=2,
        )

    if run_mode == "HARMINV" or selected_coarse_df.empty:
        summary_csv = os.path.join(base_out_dir, "summary.csv")
        selected_coarse_df.to_csv(summary_csv, index=False)
        _log(f"Pipeline finished (HARMINV mode). Summary written to {summary_csv}")
        return {
            "output_dir": base_out_dir,
            "summary_df": selected_coarse_df,
            "coarse_df": coarse_merged_df,
            "epchk_pngs": epchk_png_paths,
            "harminv_pngs": harminv_plot_paths,
            "mode_profile_pngs": mode_profile_paths,
            "derived": derived,
            "env_info": env_info,
        }

    # =========================================================================
    # STEP 2.5 (高解像度 Harminv 精密化) & STEP 3 (Running DFT 断面モードプロファイル)
    # =========================================================================
    refined_rows: List[Dict[str, Any]] = []

    for pz in parities:
        modes_in_pz = selected_coarse_df[selected_coarse_df["pz"] == pz].reset_index(drop=True)
        if modes_in_pz.empty:
            continue

        p_label = parity_label(pz)
        _log(f"--- [STEP 2.5 / 3] High-Res Mode Refinement & Profile ({p_label}, res={derived['res_mode']}) ---")

        # STEP 2.5 & STEP 3 では 4x4x4 サブピクセル平均を有効化 (仕様書B1 §7)
        eps_grid_m, coords_m, _ = build_structure_permittivity(
            cfg=cfg,
            sp=derived["sp_mode"],
            res=derived["res_mode"],
            pz=pz,
            eps_averaging=True,
            subpixel_n=int(backend.get("subpixel_n", 4)),
            anisotropic=bool(backend.get("anisotropic_averaging", False)),
        )

        epchk_hi_path = os.path.join(base_out_dir, f"epchk_highres_{p_label}.png")
        plot_epchk_slices(
            eps_grid_m,
            coords_m,
            pz=pz,
            out_png_path=epchk_hi_path,
            title=f"High-Res Permittivity (res={derived['res_mode']}, 4x4x4 subpixel)",
        )
        epchk_png_paths.append(epchk_hi_path)

        comp_cfg = derived["parity_components"].get(pz, derived["parity_components"][0])
        f_candidates = [float(r["freq"]) for _, r in modes_in_pz.iterrows()]
        f_min_cand = min(f_candidates)
        f_max_cand = max(f_candidates)
        fcen_ref = 0.5 * (f_min_cand + f_max_cand)
        # 全候補モードをカバーする狭帯域パルス幅 df_refine
        df_spread = max(f_max_cand - f_min_cand, 0.0)
        df_ref = max(df_spread * 1.5 + 0.02 * fcen_ref, fcen_ref / 100.0)
        fmin_search = max(derived["fmin"] * 0.95, f_min_cand - 0.5 * df_ref)
        fmax_search = min(derived["fmax"] * 1.05, f_max_cand + 0.5 * df_ref)

        if run_mode == "ALL":
            t0_ref = time.perf_counter()
            sim_res_ref = run_fdtdx_simulation(
                eps_grid=eps_grid_m,
                coords=coords_m,
                res=derived["res_mode"],
                fcen=fcen_ref,
                df=df_ref,
                t_max=derived["T_MAX_refine"],
                src_positions=derived["src_positions"],
                src_comps=comp_cfg["src_comps"],
                src_phases=derived["src_phases"],
                mon_positions=derived["mon_positions"],
                mon_comps=comp_cfg["mon_comps"],
                pml_cells=derived["pml_cells_mode"],
                px=derived["px"],
                py=derived["py"],
                pz=pz,
                cutoff=derived["cutoff"],
                stride=derived["stride_mode"],
                chunk_steps=int(backend.get("chunk_steps", 2000)),
                dft_freqs=f_candidates if merge_refine_mode else None,
                dtype_str=backend.get("dtype", "float32"),
                enable_x64=bool(backend.get("x64_refine", False)),
            )

            df_harm_ref = analyze_simulation_with_harminv(
                time_array=sim_res_ref.time_array,
                monitor_traces=sim_res_ref.monitor_traces,
                monitor_positions=sim_res_ref.monitor_positions,
                monitor_components=sim_res_ref.monitor_components,
                dt_rec=sim_res_ref.dt_rec,
                t_src_end=sim_res_ref.t_src_end,
                fmin=fmin_search,
                fmax=fmax_search,
                pz=pz,
            )

            sel_ref_df, _ = cluster_and_select_modes(
                df_harm_ref,
                n_modes_select=len(modes_in_pz),
                min_detections=1,
            )
            if sel_ref_df.empty:
                sel_ref_df = modes_in_pz.copy()

            w_ref = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_refine"])
            stage_key = f"STEP2.5+3_pz{pz}" if merge_refine_mode else f"STEP2.5_pz{pz}"
            estimator.finish_stage(stage_key, w_ref, time.perf_counter() - t0_ref)
            _log(
                f"STEP 2.5 finished ({p_label}): speed={sim_res_ref.mcells_per_sec:.1f} Mcells*step/s | "
                f"{estimator.format_eta()}"
            )
        else:
            sel_ref_df = modes_in_pz.copy()
            sim_res_ref = None

        # STEP 3: merge_refine_mode=False または run_mode=='MODE' の場合は確定周波数で狭帯域 DFT ランを実行
        if not merge_refine_mode or sim_res_ref is None:
            t0_m3 = time.perf_counter()
            f_final_list = [float(r["freq"]) for _, r in sel_ref_df.iterrows()]
            fcen_m3 = float(np.mean(f_final_list))
            df_m3 = max((max(f_final_list) - min(f_final_list)) * 1.4 + fcen_m3 / 200.0, fcen_m3 / 250.0)

            sim_res_dft = run_fdtdx_simulation(
                eps_grid=eps_grid_m,
                coords=coords_m,
                res=derived["res_mode"],
                fcen=fcen_m3,
                df=df_m3,
                t_max=derived["T_MAX_mode"],
                src_positions=derived["src_positions"],
                src_comps=comp_cfg["src_comps"],
                src_phases=derived["src_phases"],
                mon_positions=derived["mon_positions"],
                mon_comps=comp_cfg["mon_comps"],
                pml_cells=derived["pml_cells_mode"],
                px=derived["px"],
                py=derived["py"],
                pz=pz,
                cutoff=derived["cutoff"],
                stride=derived["stride_mode"],
                chunk_steps=int(backend.get("chunk_steps", 2000)),
                dft_freqs=f_final_list,
                dtype_str=backend.get("dtype", "float32"),
            )
            w_m3 = (float(derived["res_mode"]) ** 4) * float(derived["T_MAX_mode"])
            estimator.finish_stage(f"STEP3_pz{pz}", w_m3, time.perf_counter() - t0_m3)
            _log(
                f"STEP 3 DFT run finished ({p_label}): speed={sim_res_dft.mcells_per_sec:.1f} Mcells*step/s | "
                f"{estimator.format_eta()}"
            )
        else:
            sim_res_dft = sim_res_ref

        # 各モードの断面プロファイル画像を保存
        for local_i, (_, row) in enumerate(sel_ref_df.iterrows()):
            row_dict = row.to_dict()
            f_val = float(row_dict["freq"])
            q_val = float(row_dict["Q"])
            sym_s = str(row_dict.get("sym", p_label))
            mode_global_idx = len(refined_rows)

            if local_i in sim_res_dft.dft_slices:
                png_name = f"mode_{mode_global_idx}_f{f_val:.5f}_{sym_s}_Dden_Hden.png"
                png_path = os.path.join(base_out_dir, png_name)
                plot_mode_profile_slices(
                    dft_slice_dict=sim_res_dft.dft_slices[local_i],
                    mode_idx=mode_global_idx,
                    freq=f_val,
                    q_val=q_val,
                    sym_str=sym_s,
                    out_png_path=png_path,
                )
                mode_profile_paths.append(png_path)
                row_dict["profile_png"] = png_name

            refined_rows.append(row_dict)

    final_summary_df = pd.DataFrame(refined_rows) if refined_rows else selected_coarse_df
    if not final_summary_df.empty and "Q" in final_summary_df.columns:
        final_summary_df = final_summary_df.sort_values(by="Q", ascending=False).reset_index(drop=True)

    # 最終 best condition ファイルと summary.csv の保存
    best_final_txt, _ = save_best_condition_files(
        final_summary_df, out_dir=base_out_dir, date_str=date_str, stage_suffix=""
    )
    summary_csv_path = os.path.join(base_out_dir, "summary.csv")
    final_summary_df.to_csv(summary_csv_path, index=False)

    with open(checkpoint_path, "w", encoding="utf-8") as cf:
        json.dump(
            {
                "last_completed_step": "STEP3_DONE",
                "date_str": date_str,
                "summary_csv": summary_csv_path,
                "best_condition_txt": best_final_txt,
            },
            cf,
            indent=2,
        )

    _log(f"=== Pipeline Completed Successfully! Results saved to: {base_out_dir} ===")
    return {
        "output_dir": base_out_dir,
        "summary_df": final_summary_df,
        "coarse_df": coarse_merged_df,
        "epchk_pngs": epchk_png_paths,
        "harminv_pngs": harminv_plot_paths,
        "mode_profile_pngs": mode_profile_paths,
        "derived": derived,
        "env_info": env_info,
    }
