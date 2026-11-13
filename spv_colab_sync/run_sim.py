import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import datetime
import glob
import multiprocessing
import platform
import shutil
import subprocess
import sys
import time
import matplotlib
matplotlib.use("Agg")
import meep as mp
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    _PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    from spv_colab_sync import config as cfg_mod
    from spv_colab_sync import my_analysis as ma
    from spv_colab_sync import my_calculation_setup as mc
else:
    from . import config as cfg_mod
    from . import my_analysis as ma
    from . import my_calculation_setup as mc


def get_system_resources():
    logical_cpus = multiprocessing.cpu_count()
    phys_cores = logical_cpus
    try:
        out = subprocess.check_output(["lscpu", "-p=CORE,SOCKET"], text=True, stderr=subprocess.DEVNULL)
        core_set = {line.strip() for line in out.splitlines() if not line.startswith("#") and "," in line}
        if core_set:
            phys_cores = len(core_set)
    except Exception:
        pass

    mem_gb = 0.0
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_gb = int(line.split()[1]) / (1024.0 ** 2)
                        break
    except Exception:
        pass

    return {
        "logical_cpus": logical_cpus,
        "physical_cores": phys_cores,
        "mpi_procs": mp.count_processors() if hasattr(mp, "count_processors") else 1,
        "mem_gb": mem_gb,
        "meep_version": getattr(mp, "__version__", "unknown"),
        "platform": platform.platform(),
    }


class ProgressEstimator:
    """work = res^4 * T_MAX ベースの残り時間・終了予定時刻推定器。"""

    def __init__(self, ref_res=55):
        self.ref_res = float(max(ref_res, 1))
        self.start_wall_time = time.time()
        self.completed_work = 0.0
        self.total_work = 0.0
        self.current_task_work = 0.0
        self.current_task_raw_work = 0.0
        self.current_task_label = ""
        self.task_idx = 0
        self.total_tasks = 0
        self.sec_per_work_unit = 10.0

    def calc_raw_work(self, tmax, res):
        return (float(res) ** 4) * float(tmax)

    def calc_work(self, tmax, res):
        return self.calc_raw_work(tmax, res) / (self.ref_res ** 4)

    def set_total_plan(self, task_list):
        self.total_work = sum(self.calc_work(t, r) for t, r in task_list)
        self.total_tasks = len(task_list)

    def add_dynamic_tasks(self, task_list):
        for t, r in task_list:
            self.total_work += self.calc_work(t, r)
        self.total_tasks += len(task_list)

    def start_task(self, tmax, res, label):
        self.task_idx += 1
        self.current_task_work = self.calc_work(tmax, res)
        self.current_task_raw_work = self.calc_raw_work(tmax, res)
        self.current_task_label = f"[{self.task_idx}/{max(self.total_tasks, self.task_idx)}] {label}"
        self.task_start_time = time.time()

    def finish_task(self, actual_meep_t=None, planned_meep_t=None):
        elapsed_task = max(time.time() - self.task_start_time, 1e-3)
        self.completed_work += self.current_task_work

        eff_work = self.current_task_work
        if actual_meep_t is not None and planned_meep_t is not None and planned_meep_t > 0:
            ratio = min(max(float(actual_meep_t) / float(planned_meep_t), 0.05), 1.0)
            eff_work = self.current_task_work * ratio

        if eff_work > 0:
            measured_rate = elapsed_task / eff_work
            self.sec_per_work_unit = 0.3 * self.sec_per_work_unit + 0.7 * measured_rate

        if mp.am_master():
            rem_work = max(self.total_work - self.completed_work, 0.0)
            rem_sec = rem_work * self.sec_per_work_unit
            eta_dt = datetime.datetime.now() + datetime.timedelta(seconds=rem_sec)
            pct = min(100.0, (self.completed_work / max(self.total_work, 1e-9)) * 100.0)
            print(
                f"\n================================================================================\n"
                f"[進捗サマリー] {self.current_task_label} 完了 (所要時間: {self._fmt_sec(elapsed_task)})\n"
                f"  ・タスク仕事量   : res^4×T_MAX = {self.current_task_raw_work:.3e}\n"
                f"  ・全体進捗率     : {pct:5.1f} %\n"
                f"  ・全体残り時間   : 約 {self._fmt_sec(rem_sec)}\n"
                f"  ・全体終了予定   : {eta_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"================================================================================\n",
                flush=True
            )

    def report_in_step(self, current_meep_t, total_meep_t):
        if not mp.am_master() or total_meep_t <= 0:
            return
        frac = min(max(current_meep_t / total_meep_t, 0.0), 1.0)
        elapsed_task = time.time() - self.task_start_time

        if frac > 0.05 and self.current_task_work > 0:
            live_rate = elapsed_task / (self.current_task_work * frac)
            eff_rate = 0.5 * self.sec_per_work_unit + 0.5 * live_rate
        else:
            eff_rate = self.sec_per_work_unit

        task_rem_sec = max(self.current_task_work * (1.0 - frac) * eff_rate, 0.0)
        done_work = self.completed_work + self.current_task_work * frac
        total_rem_work = max(self.total_work - done_work, 0.0)
        total_rem_sec = total_rem_work * eff_rate

        now_dt = datetime.datetime.now()
        task_eta = now_dt + datetime.timedelta(seconds=task_rem_sec)
        total_eta = now_dt + datetime.timedelta(seconds=total_rem_sec)
        total_pct = min(100.0, (done_work / max(self.total_work, 1e-9)) * 100.0)

        print(
            f"[経過報告 {now_dt.strftime('%H:%M:%S')}] {self.current_task_label} | "
            f"ステップ進捗: {frac * 100:5.1f}% ({current_meep_t:.1f}/{total_meep_t:.1f} Meep時間) | "
            f"この計算の残り: {self._fmt_sec(task_rem_sec)} ({task_eta.strftime('%H:%M:%S')}頃) | "
            f"全体進捗: {total_pct:4.1f}% -> 全体残り: 約{self._fmt_sec(total_rem_sec)} "
            f"(終了予定: {total_eta.strftime('%m/%d %H:%M')})",
            flush=True
        )

    @staticmethod
    def _fmt_sec(sec):
        sec = int(max(sec, 0))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}時間{m:02d}分"
        return f"{m}分{s:02d}秒"


def get_latest_simcon_file(directory="./"):
    pattern = os.path.join(directory, "best condition *.txt")
    files = [f for f in glob.glob(pattern) if not f.endswith("_coarse.txt")]
    if not files:
        raise FileNotFoundError(f"'{directory}' 内に 'best condition *.txt' ファイルが見つかりません。")
    return max(files, key=os.path.getmtime)


def apply_structure(calc_obj, stype, st):
    period = float(st["period"])
    if stype == 0:
        calc_obj.set_structure(
            0,
            float(st["r_ratio"]),
            float(st["slab_ratio"]),
            int(st["Nx"]),
            int(st["Ny"]),
            int(st["defN"]),
        )
    elif stype == 3:
        calc_obj.set_structure(
            3, period, float(st["Disk_t"]), float(st["Sac_D"]), float(st["Sac_t"]), float(st["Out_D"])
        )
    elif stype == 4:
        calc_obj.set_structure(
            4, period, float(st["Disk_t"]), float(st["Sac_D"]), float(st["Sac_t"]), float(st["Out_D"]),
            int(st["Cir_reso"]), float(st["Dev_rate"])
        )
    elif stype == 5:
        calc_obj.set_structure(
            5, period, float(st["Disk_t"]), float(st["Wg_w"]), float(st["Wg_t"]), float(st["Gap"]), float(st["Out_D"])
        )
    elif stype == 6:
        calc_obj.set_structure(
            6,
            float(st["r_ratio"]),
            float(st["slab_ratio"]),
            int(st["N_ring"]),
            float(st["twist_deg"]),
            float(st["core_r_ratio"]),
            float(st["gap_ratio"]),
            float(st["bowtie_angle"]),
        )
    else:
        raise ValueError(f"未対応の structure_type です: {stype}")


def _build_summary_row(row, parity, res_val, elapsed_sec, profile_png=""):
    pz_map = {"+": 1, "-": -1, "o": 0}
    return {
        "parity": parity,
        "sym": str(row.get("sym", parity)),
        "pz": int(row.get("pz", pz_map.get(parity, 0))),
        "freq": float(row["freq"]),
        "Q": float(row["Q"]),
        "Q_max": float(row.get("Q_max", row["Q"])),
        "Q_iqr": float(row.get("Q_iqr", 0.0)),
        "|A|": float(row.get("|A|", 0.0)),
        "Re[A]": float(row.get("Re[A]", row.get("|A|", 0.0))),
        "Im[A]": float(row.get("Im[A]", 0.0)),
        "Re[Er]": float(row.get("Re[Er]", 0.0)),
        "n_detections": int(row.get("n_detections", row.get("hist n", 1))),
        "rank_score": float(row.get("rank_score", row["Q"])),
        "res": int(res_val),
        "elapsed_sec": round(float(elapsed_sec), 2),
        "excon": str(row.get("excon", "")),
        "detcon": str(row.get("detcon", "")),
        "file name": str(row.get("file name", "")),
        "profile_png": str(profile_png),
    }


def run_pipeline(user_cfg=None):
    """spv / spv_gpu と条件を揃えた MPI PyMeep + Harminv パイプラインを実行する。"""
    if user_cfg is None:
        user_cfg = cfg_mod.cfg

    norm_cfg = cfg_mod.normalize_cfg(user_cfg)
    p_derived = cfg_mod.derive_params(norm_cfg, verbose=mp.am_master())
    out_dir = os.path.abspath(p_derived["out_dir"])
    if mp.am_master():
        os.makedirs(out_dir, exist_ok=True)
    mp.all_wait()

    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    mc.my_mp_calc.run_log_path = os.path.join(out_dir, "run_log.txt")

    try:
        sys_info = get_system_resources()
        if mp.am_master():
            header_msg = (
                f"================================================================================\n"
                f"[SPV Colab Sync MPI Pipeline 起動情報]\n"
                f"  ・日時               : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  ・プラットフォーム   : {sys_info['platform']}\n"
                f"  ・CPU コア数 (nproc) : 論理 {sys_info['logical_cpus']} / 物理 {sys_info['physical_cores']} コア\n"
                f"  ・MPI プロセス数 (NP): {sys_info['mpi_procs']}\n"
                f"  ・メモリ容量 (RAM)   : {sys_info['mem_gb']:.2f} GB\n"
                f"  ・Meep バージョン    : {sys_info['meep_version']}\n"
                f"  ・出力先 (OUT_DIR)   : {out_dir}\n"
                f"--------------------------------------------------------------------------------\n"
                f"[自動導出パラメータ (spv / spv_gpu 同期条件)]\n"
                f"  ・structure_type     : {p_derived['structure_type']}\n"
                f"  ・最小特徴寸法 d_min : {p_derived['d_min']:.5f} (period={p_derived['period']}, n={p_derived['n']})\n"
                f"  ・解像度             : res_harminv={p_derived['res_harminv']}, res_mode={p_derived['res_mode']}\n"
                f"  ・周波数帯域         : f_range=({p_derived['f_min']}, {p_derived['f_max']}) -> fcen={p_derived['fcen']:.4f}, df={p_derived['df']:.4f}\n"
                f"  ・セルサイズ (粗/本) : {[round(x, 3) for x in p_derived['sp_harminv']]} / {[round(x, 3) for x in p_derived['sp_mode']]}\n"
                f"  ・dist_to_pml / PML  : {p_derived['dist_to_pml']} / {[round(x, 4) for x in p_derived['pml_w_harminv']]}\n"
                f"  ・parities           : {p_derived['parities']} | n_modes={p_derived['n_modes']} | run_mode={p_derived['run_mode']}\n"
                f"  ・時間・停止条件     : tmax_coarse={p_derived['tmax_coarse']:.1f}, use_decay_stop={p_derived['use_decay_stop']}\n"
                f"  ・高速化設定         : merge_refine={p_derived['merge_refine']}, n_shots={p_derived['n_shots']}, seed={p_derived['seed']}\n"
                f"================================================================================\n"
            )
            print(header_msg, flush=True)
            with open(mc.my_mp_calc.run_log_path, "a", buffering=1) as rlog:
                rlog.write(header_msg)

        stype = p_derived["structure_type"]
        st = p_derived["structure"]
        n = p_derived["n"]
        period = p_derived["period"]
        fcen = p_derived["fcen"]
        df = p_derived["df"]
        cutoff = p_derived["cutoff"]
        res_harminv = p_derived["res_harminv"]
        res_mode = p_derived["res_mode"]
        dist_to_pml = p_derived["dist_to_pml"]
        pml_w_harminv = p_derived["pml_w_harminv"]
        pml_w_mode = p_derived["pml_w_mode"]
        sp_harminv = p_derived["sp_harminv"]
        sp_mode = p_derived["sp_mode"]
        parities = p_derived["parities"]
        n_modes = p_derived["n_modes"]
        run_mode = p_derived["run_mode"]
        src_pos_harminv = p_derived["src_pos"]
        m_pos_harminv = p_derived["m_pos"]
        tmax_coarse = p_derived["tmax_coarse"]
        use_decay_stop = p_derived["use_decay_stop"]
        Ath = p_derived["Ath"]
        q_min = p_derived["Q_min"]
        q_expected = p_derived["Q_expected"]
        refine_tol_fac = p_derived["refine_tol_fac"]
        merge_refine = p_derived["merge_refine"]
        n_shots = p_derived["n_shots"]
        base_seed = p_derived["seed"]
        save_h5 = p_derived["save_h5"]

        do_harminv_sim = run_mode in ("ALL", "HARMINV", "HARMINV_BOTH", "HARMINV_SIM")
        do_harminv_ana = run_mode in ("ALL", "HARMINV", "HARMINV_BOTH", "HARMINV_ANA")
        do_mode_sim = run_mode in ("ALL", "MODE", "MODE_SIM")

        progress = ProgressEstimator(ref_res=res_harminv)
        planned_tasks = []
        if do_harminv_sim:
            for pz_char in parities:
                sym_combos = cfg_mod.get_symmetry_combinations(norm_cfg, pz_char)
                for _ in sym_combos:
                    for _ in range(n_shots):
                        planned_tasks.append((tmax_coarse, res_harminv))
        progress.set_total_plan(planned_tasks)

        summary_rows = []
        pipeline_t0 = time.time()
        target_simcon = None

        if do_harminv_sim or do_harminv_ana:
            mc_coarse = mc.my_mp_calc(1)
            mc_coarse.eps_averaging = False
            mc_coarse.save_h5 = save_h5
            mc_coarse.cutoff_harminv = cutoff
            mc_coarse.decay_dt = p_derived["decay_dt"]
            mc_coarse.decay_by = p_derived["decay_by"]
            mc_coarse.use_decay_stop = use_decay_stop
            mc_coarse.progress_tracker = progress

            run_date = mc_coarse.get_date()
            analysis_fname_list = []
            sim_condition = []
            coarse_elapsed_by_parity = {}

            if do_harminv_sim:
                if mp.am_master():
                    with open(f"{run_date}.txt", "w") as date_f:
                        date_f.write("#date\tparity\tshot\n")

                for pz_char in parities:
                    t_parity_start = time.time()
                    sym_combos = cfg_mod.get_symmetry_combinations(norm_cfg, pz_char)

                    for p_vec in sym_combos:
                        pz_eff_char = cfg_mod.PARITY_MAP[p_vec[2]]
                        m_field = cfg_mod.get_monitor_fields(pz_eff_char)
                        s_field = list(m_field)
                        for shot_idx in range(n_shots):
                            mc_coarse.src_seed = base_seed + shot_idx
                            mc_coarse.shot_id = shot_idx if n_shots > 1 else None
                            if mp.am_master():
                                with open(f"{run_date}.txt", "a") as date_f:
                                    date_f.write(f"{run_date}\t{pz_eff_char}\t{shot_idx}\n")

                            label = (
                                f"STEP1 同時励起Harminv(parity={pz_eff_char}, sym={p_vec}, "
                                f"res={res_harminv}, shot={shot_idx + 1}/{n_shots})"
                            )
                            progress.start_task(tmax_coarse, res_harminv, label)

                            sim_param = [
                                __file__, res_harminv, dist_to_pml, sp_harminv, pml_w_harminv,
                                p_vec, period, n, fcen, df, tmax_coarse
                            ]
                            mc_coarse.set_base_param(
                                *sim_param, src_pos_harminv, m_pos_harminv, m_field, s_field
                            )
                            apply_structure(mc_coarse, stype, st)
                            mc_coarse.start_simulation()

                            planned_meep_t = (2.0 * cutoff / df) + tmax_coarse
                            progress.finish_task(
                                actual_meep_t=mc_coarse.mpset.last_stopped_meep_time,
                                planned_meep_t=planned_meep_t
                            )

                            analysis_fname_list.extend(mc_coarse.get_file_name())
                            sim_condition.extend(mc_coarse.get_sim_condition())

                    coarse_elapsed_by_parity[pz_char] = time.time() - t_parity_start

            mp.all_wait()

            if do_harminv_ana:
                if not do_harminv_sim:
                    target_simcon = get_latest_simcon_file("./")
                    if mp.am_master():
                        print(f"[STEP 2] 既存の best condition ファイルを利用します: {target_simcon}")
                else:
                    if mp.am_master():
                        pd_list = pd.DataFrame(sim_condition, columns=mc_coarse.get_sim_condition_index())
                        pd_list["file name"] = analysis_fname_list

                        ma1 = ma.my_analysis(0)
                        ma1.set_base_param("./", pd_list, run_date)
                        ma1.start_analysis()

                        best_df = ma1.cluster_and_rank_modes(
                            Ath=Ath,
                            q_min=q_min,
                            f_range=(p_derived["f_min"], p_derived["f_max"]),
                        )
                        ma1.plot_harminv_summary(
                            selected_df=best_df.head(n_modes) if not best_df.empty else None,
                            out_png_path="harminv_summary_coarse.png",
                            f_range=(p_derived["f_min"], p_derived["f_max"]),
                        )

                        best_file = f"best condition {run_date}.txt"
                        coarse_backup = f"best condition {run_date}_coarse.txt"
                        if os.path.exists(best_file):
                            shutil.copyfile(best_file, coarse_backup)
                            target_simcon = best_file
                            print(f"[STEP 2 完了] {len(best_df)} クラスタから上位 {min(len(best_df), n_modes)} モードを対象とします: {best_file}")

                            if not do_mode_sim and not best_df.empty:
                                for _, row in best_df.head(n_modes).iterrows():
                                    pz_sym = str(row["sym"])[-1]
                                    summary_rows.append(
                                        _build_summary_row(
                                            row,
                                            parity=pz_sym,
                                            res_val=res_harminv,
                                            elapsed_sec=coarse_elapsed_by_parity.get(pz_sym, 0.0),
                                        )
                                    )
                        else:
                            print(f"[警告] Harminvの検出モードが0件でした ({run_date})。")

                    mp.all_wait()
                    if do_mode_sim:
                        try:
                            target_simcon = get_latest_simcon_file("./")
                        except FileNotFoundError:
                            target_simcon = None

        if do_mode_sim:
            if target_simcon is None:
                try:
                    target_simcon = get_latest_simcon_file("./")
                except FileNotFoundError as e:
                    if mp.am_master():
                        print(f"[スキップ] {e} Mode計算を終了します。")
                    return pd.DataFrame(summary_rows, columns=cfg_mod.SUMMARY_COLUMNS)

            st_list1_mode = pd.read_table(target_simcon)
            if "Unnamed: 0" in st_list1_mode.columns:
                st_list1_mode = st_list1_mode.drop(columns=["Unnamed: 0"])

            # spv_gpu と同様に全体ランキング上位 n_modes 件を選出
            if not st_list1_mode.empty:
                sort_cols = [c for c in ["rank_score", "Q", "|A|"] if c in st_list1_mode.columns]
                if sort_cols:
                    st_list1_mode = st_list1_mode.sort_values(
                        by=sort_cols, ascending=[False] * len(sort_cols)
                    ).iloc[:n_modes].reset_index(drop=True)

            if len(st_list1_mode) == 0:
                if mp.am_master():
                    print(f"[警告] '{target_simcon}' に有効な共振モードが存在しないため STEP 2.5 / 3 をスキップします。")
            else:
                dyn_tasks = []
                for idx in range(len(st_list1_mode)):
                    f_c = float(st_list1_mode["freq"][idx])
                    q_c = float(st_list1_mode["Q"][idx])
                    tmax_ref = cfg_mod.calc_tmax_fine(f_c, q_c, q_expected)
                    dyn_tasks.append((tmax_ref, res_mode))
                    if not merge_refine:
                        dyn_tasks.append((tmax_ref, res_mode))
                progress.add_dynamic_tasks(dyn_tasks)

                if mp.am_master():
                    backup_coarse = target_simcon.replace(".txt", "_coarse.txt")
                    if not os.path.exists(backup_coarse):
                        shutil.copyfile(target_simcon, backup_coarse)
                    print(f"\n[STEP 2.5] 解像度 res={res_mode} による狭帯域再測定を開始します (対象: {len(st_list1_mode)} モード)...")

                refined_freqs = []
                refined_Qs = []
                refined_As = []
                profile_pngs = []
                step3_completed = []
                mode_elapsed_list = []

                for idx in range(len(st_list1_mode)):
                    t_mode_start = time.time()
                    coarse_freq = float(st_list1_mode["freq"][idx])
                    coarse_Q = float(st_list1_mode["Q"][idx])
                    coarse_A = float(st_list1_mode["|A|"][idx])

                    sym_str = str(st_list1_mode["sym"][idx])
                    p_ref = [cfg_mod.PARITY_MAP[sym_str[0]], cfg_mod.PARITY_MAP[sym_str[1]], cfg_mod.PARITY_MAP[sym_str[2]]]
                    pz_char = sym_str[2]

                    detcon_parts = str(st_list1_mode["detcon"][idx]).split(",")
                    mfield_ref = [detcon_parts[0]]
                    det_pos_ratio = [float(val) / period for val in detcon_parts[1:4]]
                    m_pos_ref = [det_pos_ratio, m_pos_harminv[0]]

                    sfield_ref = cfg_mod.get_monitor_fields(pz_char)
                    df_refine = cfg_mod.calc_df_refine(coarse_freq, coarse_Q)
                    tmax_refine = cfg_mod.calc_tmax_fine(coarse_freq, coarse_Q, q_expected)

                    mc_ref = mc.my_mp_calc(1)
                    mc_ref.eps_averaging = True
                    mc_ref.save_h5 = False
                    mc_ref.cutoff_harminv = cutoff
                    mc_ref.decay_dt = p_derived["decay_dt"]
                    mc_ref.decay_by = p_derived["decay_by"]
                    mc_ref.use_decay_stop = use_decay_stop
                    mc_ref.src_seed = base_seed
                    mc_ref.capture_dft = merge_refine
                    mc_ref.dft_nfreq = 11
                    mc_ref.progress_tracker = progress

                    label_ref = (
                        f"STEP2.5 狭帯域再測定(mode={idx + 1}/{len(st_list1_mode)}, sym={sym_str}, "
                        f"res={res_mode}, f={coarse_freq:.4f}, df={df_refine:.4f}, T_MAX={tmax_refine:.0f})"
                    )
                    progress.start_task(tmax_refine, res_mode, label_ref)

                    sim_param_ref = [
                        __file__, res_mode, dist_to_pml, sp_mode, pml_w_mode, p_ref,
                        period, n, coarse_freq, df_refine, tmax_refine
                    ]
                    mc_ref.set_base_param(
                        *sim_param_ref, src_pos_harminv, m_pos_ref, mfield_ref, sfield_ref
                    )
                    apply_structure(mc_ref, stype, st)
                    mc_ref.start_simulation()

                    planned_meep_t = (2.0 * cutoff / df_refine) + tmax_refine
                    progress.finish_task(
                        actual_meep_t=mc_ref.mpset.last_stopped_meep_time,
                        planned_meep_t=planned_meep_t
                    )
                    mp.all_wait()

                    best_f, best_q, best_a = coarse_freq, coarse_Q, coarse_A
                    min_diff = 1e9
                    for fname in mc_ref.get_file_name():
                        f_list, q_list, a_list, _ = mc_ref.read_harminv_file(fname + ".txt")
                        for f_v, q_v, a_v in zip(f_list, q_list, a_list):
                            diff = abs(f_v - coarse_freq)
                            if q_v > q_min and diff < (df_refine * refine_tol_fac) and diff < min_diff:
                                min_diff = diff
                                best_f = f_v
                                best_q = q_v
                                best_a = abs(a_v)

                    refined_freqs.append(best_f)
                    refined_Qs.append(best_q)
                    refined_As.append(best_a)

                    if mp.am_master():
                        print(
                            f"[STEP 2.5 補正完了] Mode {idx + 1} ({sym_str}): "
                            f"freq {coarse_freq:.5f} (res={res_harminv}) -> {best_f:.5f} (res={res_mode}), "
                            f"Q: {coarse_Q:.1f} -> {best_q:.1f}"
                        )

                    merged_done = False
                    saved_png = ""
                    if merge_refine and best_q >= cfg_mod.Q_MERGE_THRESHOLD and abs(best_f - coarse_freq) <= 0.5 * df_refine:
                        df_mode_val = cfg_mod.calc_df_mode(best_f, best_q)
                        tmax_mode_val = cfg_mod.calc_tmax_fine(best_f, best_q, q_expected)
                        pngs = mc_ref.mpset.save_dft_mode_profile(
                            target_freq=best_f,
                            src_pos=src_pos_harminv,
                            sfield=sfield_ref,
                            df_mode=df_mode_val,
                            tmax_mode=tmax_mode_val,
                            pos_xyz=[[0], [0], [0]],
                            out_field_list=sfield_ref
                        )
                        if pngs:
                            saved_png = os.path.basename(pngs[0])
                        merged_done = True
                        if mp.am_master():
                            print(f"[STEP 2.5+3 統合完了] Mode {idx + 1}: Q={best_q:.1f} >= {cfg_mod.Q_MERGE_THRESHOLD} のため STEP 3 を省略しました。")

                    profile_pngs.append(saved_png)
                    step3_completed.append(merged_done)
                    mode_elapsed_list.append(time.time() - t_mode_start)
                    mp.all_wait()

                st_list1_mode["freq"] = refined_freqs
                st_list1_mode["Q"] = refined_Qs
                st_list1_mode["|A|"] = refined_As
                if mp.am_master():
                    st_list1_mode.to_csv(target_simcon, sep="\t", index=False)
                    print(f"[STEP 2.5] 補正後の高解像度共振条件で '{target_simcon}' を更新しました。")
                mp.all_wait()

                gif_cut_pos = [[0], [0], [0]]
                for idx in range(len(st_list1_mode)):
                    if step3_completed[idx]:
                        continue

                    t_step3_start = time.time()
                    fcen_m = float(st_list1_mode["freq"][idx])
                    q_m = float(st_list1_mode["Q"][idx])
                    sym_str = str(st_list1_mode["sym"][idx])
                    p_m = [cfg_mod.PARITY_MAP[sym_str[0]], cfg_mod.PARITY_MAP[sym_str[1]], cfg_mod.PARITY_MAP[sym_str[2]]]
                    pz_char = sym_str[2]

                    detcon_parts = str(st_list1_mode["detcon"][idx]).split(",")
                    det_pos_ratio = [float(val) / period for val in detcon_parts[1:4]]
                    sfield_m = cfg_mod.get_monitor_fields(pz_char)

                    df_mode = cfg_mod.calc_df_mode(fcen_m, q_m)
                    tmax_mode = cfg_mod.calc_tmax_fine(fcen_m, q_m, q_expected)

                    if merge_refine:
                        progress.add_dynamic_tasks([(tmax_mode, res_mode)])

                    mc_m = mc.my_mp_calc(2)
                    mc_m.eps_averaging = True
                    mc_m.save_h5 = save_h5
                    mc_m.cutoff_mode = cutoff
                    mc_m.decay_dt = p_derived["decay_dt"]
                    mc_m.decay_by = p_derived["decay_by"]
                    mc_m.use_decay_stop = use_decay_stop
                    mc_m.src_seed = base_seed
                    mc_m.decay_pt = det_pos_ratio
                    mc_m.progress_tracker = progress

                    label_m = (
                        f"STEP3 Modeプロファイル(mode={idx + 1}/{len(st_list1_mode)}, sym={sym_str}, "
                        f"res={res_mode}, f={fcen_m:.5f}, df={df_mode:.4f}, T_MAX={tmax_mode:.0f})"
                    )
                    progress.start_task(tmax_mode, res_mode, label_m)

                    sim_param_m = [
                        __file__, res_mode, dist_to_pml, sp_mode, pml_w_mode, p_m,
                        period, n, fcen_m, df_mode, tmax_mode
                    ]
                    mc_m.set_base_param(
                        *sim_param_m,
                        src_pos_harminv,
                        sfield_m,
                        sfield_m,
                        gif_cut_pos,
                        fcen_m
                    )
                    apply_structure(mc_m, stype, st)
                    mc_m.start_simulation()

                    if mc_m.mpset.last_saved_pngs:
                        profile_pngs[idx] = os.path.basename(mc_m.mpset.last_saved_pngs[0])

                    planned_meep_t = (2.0 * cutoff / df_mode) + tmax_mode
                    progress.finish_task(
                        actual_meep_t=mc_m.mpset.last_stopped_meep_time,
                        planned_meep_t=planned_meep_t
                    )
                    mode_elapsed_list[idx] += (time.time() - t_step3_start)
                    mp.all_wait()

                for idx in range(len(st_list1_mode)):
                    row = st_list1_mode.iloc[idx]
                    pz_char = str(row["sym"])[-1]
                    summary_rows.append(
                        _build_summary_row(
                            row,
                            parity=pz_char,
                            res_val=res_mode,
                            elapsed_sec=mode_elapsed_list[idx],
                            profile_png=profile_pngs[idx],
                        )
                    )

        summary_df = pd.DataFrame(summary_rows, columns=cfg_mod.SUMMARY_COLUMNS)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by="Q", ascending=False).reset_index(drop=True)

        if mp.am_master():
            summary_csv_path = os.path.join(out_dir, "summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            total_wall = time.time() - pipeline_t0
            print(
                f"\n================================================================================\n"
                f"[全パイプライン完了] 総実行時間: {total_wall:.1f} 秒\n"
                f"  ・サマリーCSV : {summary_csv_path}\n"
                f"  ・実行ログ    : {mc.my_mp_calc.run_log_path}\n"
                f"--------------------------------------------------------------------------------\n"
                f"{summary_df.to_string(index=False) if not summary_df.empty else '(検出モードなし)'}\n"
                f"================================================================================\n",
                flush=True
            )

        mp.all_wait()
        return summary_df
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    loaded_cfg = cfg_mod.load_runtime_cfg(cfg_mod.cfg)
    run_pipeline(loaded_cfg)
