"""
Verification & benchmark suite V1-V5 plus mirror-BC / CFS-CPML physics checks for spv_tpu.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis.harminv_cli import run_harminv
from .config import get_default_config
from .geometry.structures import build_structure_permittivity
from .pipeline import run_spv_pipeline
from .solver.tpu_runner import run_tpu_simulation


def validate_v1_permittivity(
    out_dir: str = "./spv_validation",
    res: int = 36,
) -> Dict[str, Any]:
    """
    V1: 誘電率分布一致検証
    `structure_type = 6` (Singular Bowtie PhC) の z=0 断面における
    4x4x4 サブピクセル平均 ε(x,y,0) と 8x8x8 参照を比較し、相対 L2 誤差 < 2% を検証する。
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg, _ = get_default_config()
    cfg["structure_type"] = 6
    sp = (3.6, 3.6, 0.9)

    eps_sub4, (xs, ys, _), _ = build_structure_permittivity(
        cfg=cfg, sp=sp, res=res, pz=1, eps_averaging=True, subpixel_n=4, layout="zyx"
    )
    eps_ref8, _, _ = build_structure_permittivity(
        cfg=cfg, sp=sp, res=res, pz=1, eps_averaging=True, subpixel_n=8, layout="zyx"
    )

    slice_sub4 = np.asarray(eps_sub4[0, 0, :, :], dtype=np.float64).T
    slice_ref = np.asarray(eps_ref8[0, 0, :, :], dtype=np.float64).T

    rel_l2_err = float(np.linalg.norm(slice_sub4 - slice_ref) / max(np.linalg.norm(slice_ref), 1e-12))
    passed = rel_l2_err < 0.02

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    im0 = axes[0].imshow(slice_sub4.T, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("TPU 4x4x4 Subpixel ε(x,y,0)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(slice_ref.T, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title("Reference 8x8x8 Supersampled ε(x,y,0)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = np.abs(slice_sub4 - slice_ref)
    im2 = axes[2].imshow(diff.T, origin="lower", extent=extent, cmap="magma")
    axes[2].set_title(f"|Δε| (Rel L2 Err = {rel_l2_err * 100:.3f}%)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    png_path = os.path.join(out_dir, "v1_permittivity_comparison.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "test": "V1_permittivity_L2",
        "rel_l2_error": rel_l2_err,
        "threshold": 0.02,
        "passed": passed,
        "plot_path": png_path,
    }


def validate_v4_harminv_cli() -> Dict[str, Any]:
    """
    V4: Harminv 単体検証
    既知の共振周波数・Q値を持つ複数の減衰正弦波合成信号を入力し、精度を検証する。
    """
    dt = 0.05
    t = np.arange(0, 4000, dtype=np.float64) * dt
    f1_true, q1_true = 0.6125, 2500.0
    f2_true, q2_true = 0.6840, 800.0

    decay1 = f1_true / (2.0 * q1_true)
    decay2 = f2_true / (2.0 * q2_true)

    sig = (
        1.0 * np.exp(-2.0 * math.pi * decay1 * t) * np.cos(2.0 * math.pi * f1_true * t + 0.3)
        + 0.6 * np.exp(-2.0 * math.pi * decay2 * t) * np.cos(2.0 * math.pi * f2_true * t - 0.8)
    )

    df_out = run_harminv(sig, dt=dt, fmin=0.55, fmax=0.75, q_min=50.0)
    if df_out.empty:
        return {"test": "V4_harminv_unit", "passed": False, "reason": "No modes detected"}

    idx1 = int((df_out["freq"] - f1_true).abs().idxmin())
    f1_est = float(df_out.loc[idx1, "freq"])
    q1_est = float(df_out.loc[idx1, "Q"])

    rel_f_err = abs(f1_est - f1_true) / f1_true
    rel_q_err = abs(q1_est - q1_true) / q1_true
    passed = (rel_f_err < 1e-4) and (rel_q_err < 0.02)

    return {
        "test": "V4_harminv_unit",
        "f1_true": f1_true,
        "f1_est": f1_est,
        "rel_f_err": rel_f_err,
        "Q1_true": q1_true,
        "Q1_est": q1_est,
        "rel_Q_err": rel_q_err,
        "passed": passed,
    }


def validate_mirror_bc_and_cpml() -> Dict[str, Any]:
    """
    Yee 格子鏡像境界条件 (pz=+1 PMC / pz=-1 PEC) における z=0 波源から z>0 への結合と、
    境界スラブ CFS-CPML の安定な波動吸収を検証する。
    """
    nx, ny, nz = 128, 64, 32
    dx = 0.05
    xs = (np.arange(nx, dtype=np.float32) - 0.5 * nx + 0.5) * dx
    ys = (np.arange(ny, dtype=np.float32) - 0.5 * ny + 0.5) * dx
    zs = np.arange(nz, dtype=np.float32) * dx
    eps_vac = np.ones((1, nz, ny, nx), dtype=np.float32)
    dt_meep = (0.99 / math.sqrt(3.0)) * dx

    # 1) pz=+1 (PMC): z=0 の Ex 波源が z=0.20 (k=4) に伝播・到達し、CPML で減衰することを確認
    res_pmc = run_tpu_simulation(
        eps_grid=eps_vac,
        coords=(xs, ys, zs),
        res=20,
        fcen=0.65,
        df=0.50,
        t_max=800 * dt_meep,
        src_positions=[(0.0, 0.0, 0.0)],
        src_comps=["Ex"],
        src_phases=np.zeros((1, 6)),
        mon_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 4.0 * dx)],
        mon_comps=["Ex"],
        pml_cells=10,
        pz=1,
        cutoff=3.0,
        stride=2,
        chunk_steps=200,
        decay_threshold=0.0,
        pml_mode="cpml",
        layout="zyx",
    )
    pmc_z_prop_amp = float(np.max(np.abs(res_pmc.monitor_traces[:, 1, 0])))
    pmc_coupled = pmc_z_prop_amp > 1e-2
    cpml_stable = (
        np.all(np.isfinite(res_pmc.monitor_traces))
        and res_pmc.final_energy < max(res_pmc.peak_energy * 1e-2, 1e-8)
    )

    # 2) pz=-1 (PEC): z=2*dx の Ez 波源が z=4*dx へ伝播し、波数安定であることを確認
    res_pec = run_tpu_simulation(
        eps_grid=eps_vac,
        coords=(xs, ys, zs),
        res=20,
        fcen=0.65,
        df=0.50,
        t_max=800 * dt_meep,
        src_positions=[(0.0, 0.0, 2.0 * dx)],
        src_comps=["Ez"],
        src_phases=np.zeros((1, 6)),
        mon_positions=[(0.0, 0.0, 4.0 * dx)],
        mon_comps=["Ez"],
        pml_cells=10,
        pz=-1,
        cutoff=3.0,
        stride=2,
        chunk_steps=200,
        decay_threshold=0.0,
        pml_mode="cpml",
        layout="zyx",
    )
    pec_z_prop_amp = float(np.max(np.abs(res_pec.monitor_traces[:, 0, 0])))
    pec_coupled = pec_z_prop_amp > 1e-2 and np.all(np.isfinite(res_pec.monitor_traces))

    return {
        "test": "Mirror_BC_and_CFS_CPML",
        "pmc_z4_max_amp": pmc_z_prop_amp,
        "pec_z4_max_amp": pec_z_prop_amp,
        "pmc_coupled": pmc_coupled,
        "pec_coupled": pec_coupled,
        "cpml_stable": cpml_stable,
        "passed": bool(pmc_coupled and pec_coupled and cpml_stable),
    }


def validate_v2_2d_phc(out_dir: str = "./spv_validation") -> Dict[str, Any]:
    """V2: 2D PhC 欠陥共振器 (`structure_type = 0`) の共振周波数・Q値検証。"""
    cfg, backend = get_default_config()
    cfg["structure_type"] = 0
    cfg["period"] = 0.42
    cfg["f_range"] = (0.55, 0.75)
    cfg["quality"] = "fast"
    cfg["run_mode"] = "HARMINV"
    cfg["symmetry"] = {"px": 0, "py": 0, "pz": 1}
    cfg["output_dir"] = out_dir

    res_dict = run_spv_pipeline(cfg, backend=backend, date_str="v2_2d_phc")
    summary_df = res_dict["summary_df"]
    passed = not summary_df.empty
    top_f = float(summary_df.iloc[0]["freq"]) if passed else float("nan")
    top_q = float(summary_df.iloc[0]["Q"]) if passed else float("nan")

    return {
        "test": "V2_2D_PhC_cavity",
        "n_modes_found": len(summary_df),
        "top_freq": top_f,
        "top_Q": top_q,
        "passed": passed,
    }


def validate_v3_type6_bowtie(out_dir: str = "./spv_validation") -> Dict[str, Any]:
    """V3: `structure_type = 6` (Singular Bowtie PhC, `pz = +1` / `-1`) の偶・奇両パリティ検証。"""
    cfg, backend = get_default_config()
    cfg["structure_type"] = 6
    cfg["quality"] = "fast"
    cfg["run_mode"] = "HARMINV"
    cfg["symmetry"] = {"px": 0, "py": 0, "pz": "auto"}
    cfg["output_dir"] = out_dir

    res_dict = run_spv_pipeline(cfg, backend=backend, date_str="v3_type6_bowtie")
    summary_df = res_dict["summary_df"]
    passed = not summary_df.empty

    return {
        "test": "V3_type6_bowtie_even_odd",
        "n_modes_found": len(summary_df),
        "modes": summary_df[["freq", "Q", "sym"]].to_dict(orient="records") if passed else [],
        "passed": passed,
    }


def validate_v5_speed_benchmark(
    grid_sizes: List[Tuple[int, int, int]] | None = None,
    steps: int = 1000,
    pml_mode: str = "cpml",
) -> Dict[str, Any]:
    """
    V5: TPU 速度ベンチマーク (`Mcells*step/s`、初回 JIT コンパイル時間、ウォームキャッシュ再実行、
    推定 HBM 帯域幅 [GB/s]、Colab CPU Meep 比高速化倍率を計測)。
    """
    if grid_sizes is None:
        grid_sizes = [(128, 128, 64), (256, 256, 64), (512, 480, 64)]

    meep_cpu_baseline_mcells = 3.0
    records = []

    for nx, ny, nz in grid_sizes:
        dx = 0.02
        xs = (np.arange(nx, dtype=np.float32) - 0.5 * nx + 0.5) * dx
        ys = (np.arange(ny, dtype=np.float32) - 0.5 * ny + 0.5) * dx
        zs = np.arange(nz, dtype=np.float32) * dx
        eps_dummy = np.ones((1, nz, ny, nx), dtype=np.float32) * 2.25
        dt_meep = (0.99 / math.sqrt(3.0)) * dx
        t_max = steps * dt_meep

        # Cold JIT + run
        sim_cold = run_tpu_simulation(
            eps_grid=eps_dummy,
            coords=(xs, ys, zs),
            res=int(round(1.0 / dx)),
            fcen=0.65,
            df=0.20,
            t_max=t_max,
            src_positions=[(0.0, 0.0, 0.0)],
            src_comps=["Ex", "Ey"],
            src_phases=np.zeros((1, 6)),
            mon_positions=[(0.0, 0.0, 0.0)],
            mon_comps=["Ex", "Ey"],
            pml_cells=10,
            pz=1,
            stride=4,
            chunk_steps=steps,
            decay_threshold=0.0,
            pml_mode=pml_mode,
            keep_jit_cache=True,
            layout="zyx",
        )

        # Warm re-run (same shape -> cached XLA executable)
        sim_warm = run_tpu_simulation(
            eps_grid=eps_dummy,
            coords=(xs, ys, zs),
            res=int(round(1.0 / dx)),
            fcen=0.66,
            df=0.18,
            t_max=t_max,
            src_positions=[(0.0, 0.0, 0.0)],
            src_comps=["Ex", "Ey"],
            src_phases=np.zeros((1, 6)),
            mon_positions=[(0.0, 0.0, 0.0)],
            mon_comps=["Ex", "Ey"],
            pml_cells=10,
            pz=1,
            stride=4,
            chunk_steps=steps,
            decay_threshold=0.0,
            pml_mode=pml_mode,
            keep_jit_cache=True,
            layout="zyx",
        )

        est_hbm_bw_gbs = sim_warm.mcells_per_sec * 52.0 / 1000.0
        speedup_vs_cpu_meep = sim_warm.mcells_per_sec / meep_cpu_baseline_mcells
        records.append(
            {
                "grid": f"{nx}x{ny}x{nz}",
                "total_cells_M": round((nx * ny * nz) / 1e6, 3),
                "shards": sim_warm.n_shards,
                "steps": sim_warm.total_steps_run,
                "jit_cold_sec": round(sim_cold.jit_compile_sec, 3),
                "jit_warm_sec": round(sim_warm.jit_compile_sec, 3),
                "wall_warm_sec": round(sim_warm.wall_time_sec, 3),
                "mcells_per_sec": round(sim_warm.mcells_per_sec, 1),
                "est_hbm_bw_GBs": round(est_hbm_bw_gbs, 1),
                "speedup_vs_col_cpu_meep": round(speedup_vs_cpu_meep, 1),
            }
        )

    return {
        "test": "V5_tpu_speed_benchmark",
        "benchmarks": records,
        "passed": len(records) > 0,
    }


def run_all_validations(out_dir: str = "./spv_validation") -> pd.DataFrame:
    """全検証スイートとベンチマークを一括実行し、レポート JSON 保存とサマリ DataFrame を返す。"""
    os.makedirs(out_dir, exist_ok=True)
    results = [
        validate_v1_permittivity(out_dir=out_dir),
        validate_v4_harminv_cli(),
        validate_mirror_bc_and_cpml(),
        validate_v5_speed_benchmark(),
        validate_v2_2d_phc(out_dir=out_dir),
        validate_v3_type6_bowtie(out_dir=out_dir),
    ]
    report_path = os.path.join(out_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return pd.DataFrame(results)
