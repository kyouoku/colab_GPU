"""
Notebook-facing orchestration helpers for spv_colab_sync.
Launches MPI PyMeep runs via `mpirun` and returns structured results for display.
"""

from __future__ import annotations

import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from . import config as cfg_mod


def detect_host_resources(verbose: bool = True) -> Tuple[int, int, float]:
    """物理コア数 (NP推奨)・論理CPU数・メモリ容量を検出する。"""
    logical_cpus = multiprocessing.cpu_count()
    phys_cores = logical_cpus
    cpu_model = platform.processor() or "Unknown CPU"
    try:
        lscpu_out = subprocess.check_output(["lscpu"], text=True)
        for line in lscpu_out.splitlines():
            if "Model name:" in line:
                cpu_model = line.split(":", 1)[1].strip()
        p_out = subprocess.check_output(["lscpu", "-p=CORE,SOCKET"], text=True)
        core_set = {line.strip() for line in p_out.splitlines() if not line.startswith("#") and "," in line}
        if core_set:
            phys_cores = len(core_set)
    except Exception:
        phys_cores = max(1, logical_cpus // 2) if logical_cpus > 1 else 1

    mem_gb = 0.0
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_gb = int(line.split()[1]) / (1024 ** 2)
                    break

    if verbose:
        print("=" * 72)
        print(f"CPU モデル          : {cpu_model}")
        print(f"論理 CPU 数 (nproc) : {logical_cpus}")
        print(f"物理コア数 (NP推奨) : {phys_cores} (HT除外)")
        print(f"メモリ容量 (RAM)    : {mem_gb:.2f} GB")
        print("=" * 72)
        if phys_cores < 4:
            print("[情報] 物理コア数が少ない環境です。TPUホストまたはハイメモリCPU ランタイムを推奨します。")

    return phys_cores, logical_cpus, mem_gb


def _resolve_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_outputs(out_dir: str) -> Dict[str, Any]:
    """出力ディレクトリから summary.csv, best condition, PNG 一覧を収集する。"""
    out_dir = os.path.abspath(out_dir)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()

    best_files = sorted(
        [f for f in glob.glob(os.path.join(out_dir, "best condition *.txt")) if not f.endswith("_coarse.txt")],
        key=os.path.getmtime,
    )
    latest_best = best_files[-1] if best_files else None
    coarse_best = latest_best.replace(".txt", "_coarse.txt") if latest_best else None

    best_df = pd.read_table(latest_best) if latest_best and os.path.exists(latest_best) else pd.DataFrame()
    coarse_df = pd.read_table(coarse_best) if coarse_best and os.path.exists(coarse_best) else pd.DataFrame()

    return {
        "output_dir": out_dir,
        "summary_csv": summary_csv if os.path.exists(summary_csv) else None,
        "summary_df": summary_df,
        "best_txt": latest_best,
        "best_df": best_df,
        "coarse_txt": coarse_best if coarse_best and os.path.exists(coarse_best) else None,
        "coarse_df": coarse_df,
        "epchk_pngs": sorted(glob.glob(os.path.join(out_dir, "epchk*.png"))),
        "harminv_pngs": sorted(glob.glob(os.path.join(out_dir, "harminv_summary*.png"))),
        "mode_profile_pngs": sorted(glob.glob(os.path.join(out_dir, "M(*).png"))),
    }


def _resolve_mpirun_executable() -> str:
    for cand in ("/usr/local/bin/mpirun", shutil.which("mpirun"), "mpirun"):
        if cand and os.path.exists(cand):
            return cand
    return shutil.which("mpirun") or "mpirun"


def _build_mpirun_prefix(np_procs: int) -> List[str]:
    """MPICH (Hydra) と OpenMPI の双方に対応する mpirun コマンド先頭部を構築する。"""
    mpirun_bin = _resolve_mpirun_executable()
    supports_allow_root = False
    try:
        res_ver = subprocess.run(
            [mpirun_bin, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        ver_text = ((res_ver.stdout or "") + "\n" + (res_ver.stderr or "")).lower()
        if "open mpi" in ver_text or "open-mpi" in ver_text:
            supports_allow_root = True
        elif "hydra" not in ver_text and "mpich" not in ver_text:
            res_help = subprocess.run(
                [mpirun_bin, "--help"], capture_output=True, text=True, timeout=5, check=False
            )
            help_text = (res_help.stdout or "") + "\n" + (res_help.stderr or "")
            supports_allow_root = "--allow-run-as-root" in help_text
    except Exception:
        supports_allow_root = False

    prefix = [mpirun_bin]
    if supports_allow_root:
        prefix.append("--allow-run-as-root")
    prefix.extend(["-np", str(int(np_procs))])
    return prefix


def _resolve_mpi_python() -> str:
    """condacolab 等で PyMeep が導入されている Python 実行ファイルを優先して選択する。"""
    candidates: List[str] = []
    for cand in (shutil.which("python"), "/usr/local/bin/python", shutil.which("python3"), sys.executable):
        if cand and cand not in candidates:
            candidates.append(cand)

    for py_exe in candidates:
        try:
            res = subprocess.run(
                [py_exe, "-c", "import meep"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode == 0:
                return py_exe
        except Exception:
            continue
    return sys.executable


def run_mpi_pipeline(
    cfg: Optional[Dict[str, Any]] = None,
    np_procs: Optional[int] = None,
    stream_output: bool = True,
) -> Dict[str, Any]:
    """
    設定辞書 `cfg` を保存し、`mpirun` 経由で `spv_colab_sync.run_sim` を並列実行する。
    """
    norm_cfg = cfg_mod.normalize_cfg(cfg)
    derived = cfg_mod.derive_params(norm_cfg, verbose=False)
    out_dir = os.path.abspath(derived["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    if np_procs is None:
        env_np = os.environ.get("NP")
        if env_np:
            np_procs = int(env_np)
        else:
            np_procs, _, _ = detect_host_resources(verbose=False)

    cfg_json_path = cfg_mod.save_runtime_cfg(norm_cfg, os.path.join(out_dir, "spv_runtime_cfg.json"))
    repo_root = _resolve_repo_root()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["SPV_OUT_DIR"] = out_dir
    env["SPV_USER_CFG_JSON"] = cfg_json_path
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{prev_pp}" if prev_pp else repo_root

    py_exe = _resolve_mpi_python()
    cmd = _build_mpirun_prefix(int(np_procs)) + [
        py_exe,
        "-u",
        "-m",
        "spv_colab_sync.run_sim",
    ]
    if stream_output:
        print(f"[実行コマンド] {' '.join(cmd)}")
        print(f"[保存先]       {out_dir}")
        print("-" * 80)

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if stream_output:
            print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"MPI シミュレーションがエラー終了しました (exit code={ret})")

    result = collect_outputs(out_dir)
    result["derived"] = derived
    return result
