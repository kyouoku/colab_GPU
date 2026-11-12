import os
# MPI並列時のスレッド競合（オーバーサブスクリプション）を防止
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import datetime
import glob
import math
import re
import shutil
import sys
import time
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import my_analysis as ma
import my_calculation_setup as mc


class ProgressEstimator:
    """
    解像度の違い ((res / ref_res)^4) を考慮して全体の仕事量を換算し、
    計算途中の残り時間と終了予定時刻 (ETA) をリアルタイム推定するクラス。
    """
    def __init__(self, ref_res=55):
        self.ref_res = float(ref_res)
        self.start_wall_time = time.time()
        self.completed_work = 0.0
        self.total_work = 0.0
        self.current_task_work = 0.0
        self.current_task_label = ""
        self.task_idx = 0
        self.total_tasks = 0
        # 初期推定値: ref_res=55 における 1 Meep時間あたりの秒数 (実測値で順次自動補正)
        self.sec_per_work_unit = 18.0

    def calc_work(self, meep_time, res):
        return float(meep_time) * ((float(res) / self.ref_res) ** 4)

    def set_total_plan(self, task_list):
        """task_list: [(meep_time, res), ...]"""
        self.total_work = sum(self.calc_work(t, r) for t, r in task_list)
        self.total_tasks = len(task_list)

    def add_dynamic_tasks(self, task_list):
        """STEP 2の結果に応じて動的に決まるSTEP 2.5 / STEP 3のタスクを追加"""
        for t, r in task_list:
            self.total_work += self.calc_work(t, r)
        self.total_tasks += len(task_list)

    def start_task(self, meep_time, res, label):
        self.task_idx += 1
        self.current_task_work = self.calc_work(meep_time, res)
        self.current_task_label = f"[{self.task_idx}/{max(self.total_tasks, self.task_idx)}] {label}"
        self.task_start_time = time.time()

    def finish_task(self):
        elapsed_task = max(time.time() - self.task_start_time, 1e-3)
        self.completed_work += self.current_task_work
        if self.current_task_work > 0:
            measured_rate = elapsed_task / self.current_task_work
            # 実測速度で推定レートを平滑化更新
            self.sec_per_work_unit = 0.3 * self.sec_per_work_unit + 0.7 * measured_rate

        if mp.am_master():
            rem_work = max(self.total_work - self.completed_work, 0.0)
            rem_sec = rem_work * self.sec_per_work_unit
            eta_dt = datetime.datetime.now() + datetime.timedelta(seconds=rem_sec)
            pct = min(100.0, (self.completed_work / max(self.total_work, 1e-9)) * 100.0)
            print(
                f"\n================================================================================\n"
                f"[進捗サマリー] {self.current_task_label} 完了 (所要時間: {self._fmt_sec(elapsed_task)})\n"
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

        # タスク内の進捗が5%を超えたら実測速度をリアルタイム反映
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


def get_latest_date_file(directory='./'):
    """最新の YYYYMMDD-HHMMSS.txt ログファイルを自動検出する"""
    pattern = os.path.join(directory, '[0-9]' * 8 + '-' + '[0-9]' * 6 + '.txt')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("日付ファイル (YYYYMMDD-HHMMSS.txt) が見つかりません。")
    latest_file = max(files, key=os.path.getmtime)
    return os.path.splitext(os.path.basename(latest_file))[0]


def get_latest_simcon_file(directory='./'):
    """最新の 'best condition *.txt' ファイルを自動検出する"""
    pattern = os.path.join(directory, 'best condition *.txt')
    files = [f for f in glob.glob(pattern) if not f.endswith('_coarse.txt')]
    if not files:
        raise FileNotFoundError("'best condition *.txt' ファイルが見つかりません。")
    return max(files, key=os.path.getmtime)


if __name__ == '__main__':
    # ==========================================================================
    # ★ 1. 実行モード設定 (RUN_MODE)
    # ==========================================================================
    # 'ALL'          : [全自動] ①Harminv粗探索 ➔ ②解析 ➔ ②.5高解像度補正 ➔ ③Mode計算
    # 'HARMINV_BOTH' : [半自動] ①Harminv粗探索 ➔ ②解析 までを連続実行
    # 'HARMINV_SIM'  : [個別]   ①Harminv粗探索のみ実行
    # 'HARMINV_ANA'  : [個別]   ②Harminv解析のみ実行
    # 'MODE_SIM'     : [個別]   ②.5高解像度補正 ➔ ③Mode計算を実行
    # ==========================================================================
    RUN_MODE = 'ALL'

    af_date = 'AUTO'
    simcon_file = 'AUTO'
    home_dir = './'

    # 高速化・手法B用の制御オプション
    ENABLE_STEP2_5_REFINE = True     # 手法B: STEP 2.5 で高解像度ピンポイント周波数補正を行う
    MAX_MODES_FOR_MODE_CALC = 2      # STEP 2.5 & STEP 3 で計算する最大モード数（Q値上位2件に絞り高速化）
    SAVE_H5_FILES = False            # 不要な大容量 .h5 ファイルのディスク書き出しを省略して高速化

    # ==========================================================================
    # ★ 2. 構造パラメータ (Structure Params: μm)
    # ==========================================================================
    structure_type = 6

    # --- 誘電体ボウタイ統合型2次元ツイスト格子ナノ共振器 (structure_type = 6) 用パラメータ ---
    n_phc = 3.4
    a_period = 0.45
    r_ratio = 0.26
    slab_ratio = 0.50
    N_ring = 8
    twist_deg = 3.5
    core_r_ratio = 0.55
    gap_ratio = 0.02
    bowtie_angle = 60.0

    # --- ディスク・リング共振器 (structure_type = 3 ~ 5) 用パラメータ ---
    n_disk = 2.4
    period_disk = 1.5
    Disk_R = 1.0
    w = 0.3
    Disk_t = 0.14
    Sac_t = 0.5
    Sac_D = 0
    Out_D = 10
    Gap = 0.12
    Wg_w = 0.155
    Wg_t = 0.14
    Cir_reso = 50 * 4
    Dev_rate = 0 / period_disk * 2

    # ==========================================================================
    # ★ 3. シミュレーション共通条件・高速化パラメータ
    # ==========================================================================
    parity = [-1, 1, 0]
    p_dict = {1: '+', -1: '-', 0: 'o', '+': 1, '-': -1, 'o': 0}
    field_com = ['hx', 'hy', 'hz', 'ex', 'ey', 'ez']
    m_field = ['hx', 'hy', 'hz', 'ex', 'ey', 'ez']

    cutoff_harminv = 4.0  # Harminv波源のcutoff (デフォルト5.0から短縮)
    cutoff_mode = 3.0     # Mode計算波源のcutoff (デフォルト5.0から3.0へ40%短縮)

    if structure_type == 6:
        n = n_phc
        period = a_period
        slab_t = round(slab_ratio * period, 3)

        # ★ 手法B: 2段階解像度設定
        res_harminv = 55                 # STEP 1 (Harminv粗探索用): 速度と信頼性の最適バランス値
        res_mode = 80                    # STEP 2.5 & STEP 3 (本番用): 高精度解像度
        resolution = res_harminv

        # ★ 分類A-①: 波源帯域幅と減衰待ち時間の適正化
        df = 0.20                        # 広帯域1パルスで 0.55〜0.75 を一括カバー
        df_refine = 0.06                 # STEP 2.5 ピンポイント周波数補正用の帯域幅 (0.04 ➔ 0.06でパルス短縮)
        tmax_refine = 180                # STEP 2.5 ピンポイント周波数補正用の減衰待ち時間 (250 ➔ 180に短縮)
        df_mode = 0.025                  # STEP 3 Mode計算用の狭帯域幅 (0.012 ➔ 0.025でパルス長を500➔240に短縮)

        # ★ 分類A-③: 余白 (dist_to_pml) をスリム化し、PMLが占める計算体積を約20〜25%削減
        dist_to_pml = [0.35, 0.35, 0.55]
        pml_w = [0.35, 0.35, 0.35]

        xy_span = 2 * N_ring * period + 2 * dist_to_pml[0]
        zm = slab_t + 2 * dist_to_pml[2]
        sp = [xy_span, xy_span, zm]

        # ★ Harminv探索ループ (6回)
        st_list1_harminv = [0]
        st_list2_harminv = [0]
        t_list_harminv = [300]
        f_list_harminv = [0.65]

        px_indices = [2]
        py_indices = [2]
        pz_fixed = 1

        # ★ 分類A-④: pz_fixed = 1 (偶対称) で z=0 面上にて非ゼロとなる3成分のみを監視し、Harminv負荷とI/Oを半減
        m_field = ['ex', 'ey', 'hz']

        # ギャップ中心 [0,0,0] と非対称モードも確実に拾う近傍点 [0.05, 0.05, 0.0] の代表2点
        src_pos_harminv = np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.0]
        ])
        src_field_harminv = ['ex', 'ey', 'hz']

        m_pos_harminv = [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.15, 0.10, 0.0],
            [0.0, 0.0, 0.25 * slab_ratio],
        ]

        # ループ回数(6回)に合わせてヒストグラム閾値 nth を調整
        analysis_flag = 0
        n_th = 2
        max_cons = 1
        sort_basis = 2
        bin_num = 300
        nth = 2
        Ath = 1e-5

        # ★ 分類A-①: Mode計算の減衰待ち時間を 300 ➔ 200 に短縮
        t_list_mode = [200]
        m_pos_mode = [[0.0, 0.0, 0.0]]
        gif_cut_pos = [[0], [0], [0]]

    else:
        n = n_disk
        period = period_disk
        df = 0.1
        df_refine = 0.04
        tmax_refine = 300
        df_mode = df
        res_harminv = 60
        res_mode = 90
        resolution = res_harminv
        dist_to_pml = [0.7, 0.7, 0.6]
        pml_w = [0.1, 0.1, 0.1]
        sp = [period + dist_to_pml[0], period + dist_to_pml[1], Disk_t + dist_to_pml[2]]

        st_list1_harminv = [0]
        st_list2_harminv = [0]
        t_list_harminv = np.arange(400, 901, 900)
        f_list_harminv = np.arange(2.23, 2.31, 0.05)

        px_indices = range(0, 2)
        py_indices = range(1, 2)
        pz_fixed = 0

        src_pos_harminv = np.array([
            [0.2, 0, 0 * Disk_t / period],
            [0.2, 0, -0.45 * Disk_t / period],
            [0, 0.2, 0 * Disk_t / period],
            [0, 0.2, -0.45 * Disk_t / period]
        ])
        src_field_harminv = ['hx', 'hy', 'hz']
        m_pos_harminv = [
            [0, 0.475, 0 * Disk_t / period],
            [0, 0.475, 0.45 * Disk_t / period],
            [Disk_t / period / 2, 0.475, 0 * Disk_t / period],
            [Disk_t / period / 2, 0.475, 0.45 * Disk_t / period],
            [Disk_t / period, 0.475, 0 * Disk_t / period],
            [Disk_t / period, 0.475, 0.45 * Disk_t / period],
            [0.475, 0, 0 * Disk_t / period],
            [0.475, 0, 0.45 * Disk_t / period],
            [0.475, Disk_t / period / 2, 0 * Disk_t / period],
            [0.475, Disk_t / period / 2, 0.45 * Disk_t / period],
            [0.475, Disk_t / period, 0 * Disk_t / period],
            [0.475, Disk_t / period, 0.45 * Disk_t / period],
        ]

        analysis_flag = 0
        n_th = 150
        max_cons = 1
        sort_basis = 2
        bin_num = 500
        nth = 150
        Ath = 1e-5

        t_list_mode = np.arange(10, 901, 900)
        m_pos_mode = [[-0.0, -0.475, 0.0 * Disk_t / period]]
        gif_cut_pos = [[0], [0], [0]]

    def apply_structure(calc_obj):
        if structure_type == 3:
            calc_obj.set_structure(3, period, Disk_t, Sac_D, Sac_t, Out_D)
        elif structure_type == 4:
            calc_obj.set_structure(4, period, Disk_t, Sac_D, Sac_t, Out_D, Cir_reso, Dev_rate)
        elif structure_type == 5:
            calc_obj.set_structure(5, period, Disk_t, Wg_w, Wg_t, Gap, Out_D)
        elif structure_type == 6:
            calc_obj.set_structure(
                6, r_ratio, slab_ratio, N_ring, twist_deg, core_r_ratio, gap_ratio, bowtie_angle
            )
        else:
            raise ValueError(f"未対応の structure_type です: {structure_type}")

    do_harminv_sim = RUN_MODE in ('ALL', 'HARMINV_BOTH', 'HARMINV_SIM')
    do_harminv_ana = RUN_MODE in ('ALL', 'HARMINV_BOTH', 'HARMINV_ANA')
    do_mode_sim    = RUN_MODE in ('ALL', 'MODE_SIM')

    # ==========================================================================
    # ★ 進捗・残り時間推定器の初期化
    # ==========================================================================
    progress = ProgressEstimator(ref_res=res_harminv)
    planned_tasks = []
    extra_mode_t = (2.0 / res_mode) if SAVE_H5_FILES else 0.0

    if do_harminv_sim:
        n_harminv_runs = (
            len(st_list1_harminv) * len(st_list2_harminv) *
            len(f_list_harminv) * len(px_indices) * len(py_indices) *
            len(src_pos_harminv) * len(src_field_harminv)
        )
        for tmax in t_list_harminv:
            meep_t = (2.0 * cutoff_harminv / df) + tmax
            for _ in range(n_harminv_runs):
                planned_tasks.append((meep_t, res_harminv))

    if do_mode_sim and do_harminv_sim:
        # 初期見積もりとして想定モード数 (MAX_MODES_FOR_MODE_CALC) 分を仮登録
        est_modes = MAX_MODES_FOR_MODE_CALC
        if ENABLE_STEP2_5_REFINE:
            meep_t_ref = (2.0 * cutoff_harminv / df_refine) + tmax_refine
            for _ in range(est_modes):
                planned_tasks.append((meep_t_ref, res_mode))
        for tmax in t_list_mode:
            meep_t_mode = (2.0 * cutoff_mode / df_mode) + tmax + extra_mode_t
            for _ in range(est_modes):
                planned_tasks.append((meep_t_mode, res_mode))

    progress.set_total_plan(planned_tasks)

    # ==========================================================================
    # STEP 1 & 2: Harminv シミュレーション (粗探索 res=55) & データ解析
    # ==========================================================================
    generated_simcon_files = []

    if do_harminv_sim or do_harminv_ana:
        flag = 1
        mc1 = []
        date_index = -1
        recorded_sim_dates = []
        resolution = res_harminv  # STEP 1 は探索用解像度で実行

        if do_harminv_sim:
            mc_init = mc.my_mp_calc(flag)
            run_date = mc_init.get_date()
            if mp.am_master():
                with open(os.path.join(home_dir, f'{run_date}.txt'), 'w') as date_f:
                    date_f.write('#date\tstlist1\tst_list2\n')
        else:
            if af_date == 'AUTO':
                af_date = get_latest_date_file(home_dir)
                if mp.am_master():
                    print(f"[AUTO] 最新のログファイルを読み込みます: {af_date}.txt")
            date_list = pd.read_csv(os.path.join(home_dir, f'{af_date}.txt'), sep="\t").iloc[:, 0].to_list()

        for st1 in range(len(st_list1_harminv)):
            for st2 in range(len(st_list2_harminv)):
                date_index += 1
                mc_inst = mc.my_mp_calc(flag)
                # STEP 1はサブピクセル平滑化をOFFにして初期化を高速化
                mc_inst.eps_averaging = False
                mc_inst.save_h5 = SAVE_H5_FILES
                mc_inst.cutoff_harminv = cutoff_harminv
                mc_inst.progress_tracker = progress
                mc1.append(mc_inst)
                sim_date = mc1[date_index].get_date()

                if do_harminv_sim:
                    recorded_sim_dates.append(sim_date)
                    if mp.am_master():
                        with open(os.path.join(home_dir, f'{run_date}.txt'), mode='a') as date_f:
                            date_f.write(f"{sim_date}\t{st_list1_harminv[st1]:.3f}\t{st_list2_harminv[st2]:.3f}\n")

                    # --- STEP 1: Harminv FDTD シミュレーション実行 ---
                    for tmax in t_list_harminv:
                        for fcen in f_list_harminv:
                            for i in px_indices:
                                for j in py_indices:
                                    p = [parity[i], parity[j], pz_fixed]
                                    for sn in range(len(src_pos_harminv)):
                                        for l in range(len(src_field_harminv)):
                                            meep_t = (2.0 * cutoff_harminv / df) + tmax
                                            label = (
                                                f"STEP1 Harminv(res={resolution}, f={fcen:.2f}, "
                                                f"src={src_field_harminv[l]}@{src_pos_harminv[sn][:2]})"
                                            )
                                            progress.start_task(meep_t, resolution, label)

                                            sim_param = [__file__, resolution, dist_to_pml, sp, pml_w, p, period, n, fcen, df, tmax]
                                            mc1[date_index].set_base_param(
                                                *sim_param,
                                                src_pos_harminv[sn],
                                                m_pos_harminv,
                                                m_field,
                                                src_field_harminv[l]
                                            )
                                            apply_structure(mc1[date_index])
                                            mc1[date_index].start_simulation()
                                            progress.finish_task()

                mp.all_wait()

                # --- STEP 2: Harminv データ解析実行 ---
                if do_harminv_ana:
                    if do_harminv_sim:
                        current_af_date = sim_date
                    else:
                        current_af_date = date_list[date_index] if len(date_list) != 0 else af_date

                    analysis_fname_list = []
                    sim_condition = []

                    # ★ 分類A-⑤: コピーコンストラクタ (mc2) による重複した構造生成を排除
                    mc1[date_index].date = current_af_date

                    for tmax in t_list_harminv:
                        for fcen in f_list_harminv:
                            for i in px_indices:
                                for j in py_indices:
                                    p = [parity[i], parity[j], pz_fixed]
                                    for sn in range(len(src_pos_harminv)):
                                        for l in range(len(src_field_harminv)):
                                            sim_param = [__file__, resolution, dist_to_pml, sp, pml_w, p, period, n, fcen, df, tmax]
                                            mc1[date_index].set_base_param(
                                                *sim_param,
                                                src_pos_harminv[sn],
                                                m_pos_harminv,
                                                m_field,
                                                src_field_harminv[l]
                                            )
                                            apply_structure(mc1[date_index])
                                            analysis_fname_list.extend(mc1[date_index].get_file_name())
                                            sim_condition.extend(mc1[date_index].get_sim_condition())

                    if mp.am_master():
                        pd_list = pd.DataFrame(sim_condition, columns=mc1[date_index].get_sim_condition_index())
                        pd_list['file name'] = analysis_fname_list

                        ma1 = ma.my_analysis(analysis_flag)
                        ma1.set_base_param(home_dir, pd_list, current_af_date)
                        newpd_list = ma1.start_analysis()

                        if not newpd_list.empty:
                            ma1.hist_analysis(n_th, max_cons, sort_basis, bin_num)
                            ma1.correlated_simco(nth, Ath, bin_num)
                            best_file = f'best condition {current_af_date}.txt'
                            if os.path.exists(best_file):
                                generated_simcon_files.append(best_file)
                                print(f"[完了] Best condition ファイルを出力しました: {best_file}")
                        else:
                            print(f"[警告] Harminvの検出モードが0件のため、ヒストグラム解析をスキップしました ({current_af_date})。")

                    mp.all_wait()

    # ==========================================================================
    # STEP 2.5 (高解像度ピンポイント周波数補正) & STEP 3 (Mode プロファイル計算)
    # ==========================================================================
    if do_mode_sim:
        resolution = res_mode  # ★本番用の高解像度 (80) へ引き上げ

        try:
            if simcon_file == 'AUTO':
                target_simcon = get_latest_simcon_file('./')
            else:
                target_simcon = simcon_file
        except FileNotFoundError as e:
            if mp.am_master():
                print(f'[スキップ] {e} Mode計算を終了します。')
            sys.exit(0)

        st_list1_mode = pd.read_table(target_simcon)

        # 計算時間短縮のため、Q値が高い上位 MAX_MODES_FOR_MODE_CALC 件に絞り込み
        if len(st_list1_mode) > MAX_MODES_FOR_MODE_CALC:
            st_list1_mode = (
                st_list1_mode.sort_values(by='Q', ascending=False)
                .iloc[:MAX_MODES_FOR_MODE_CALC]
                .sort_values(by='freq', ascending=True)
                .reset_index(drop=True)
            )
            if mp.am_master():
                print(f"[情報] 抽出モード数が多いため、Q値上位 {MAX_MODES_FOR_MODE_CALC} モードに絞って高解像度計算を行います。")

        # 単独で MODE_SIM を動かした場合の進捗計画の登録
        if not do_harminv_sim and len(st_list1_mode) > 0:
            dyn_tasks = []
            if ENABLE_STEP2_5_REFINE:
                meep_t_ref = (2.0 * cutoff_harminv / df_refine) + tmax_refine
                for _ in range(len(st_list1_mode)):
                    dyn_tasks.append((meep_t_ref, res_mode))
            for tmax in t_list_mode:
                meep_t_mode = (2.0 * cutoff_mode / df_mode) + tmax + extra_mode_t
                for _ in range(len(st_list1_mode)):
                    dyn_tasks.append((meep_t_mode, res_mode))
            progress.set_total_plan(dyn_tasks)

        # ----------------------------------------------------------------------
        # ★ STEP 2.5: 高解像度 (res=80) でのピンポイント周波数補正 (手法B)
        # ----------------------------------------------------------------------
        if ENABLE_STEP2_5_REFINE and len(st_list1_mode) > 0:
            if mp.am_master():
                backup_coarse = target_simcon.replace('.txt', '_coarse.txt')
                shutil.copyfile(target_simcon, backup_coarse)
                print(f"\n[STEP 2.5] 解像度 res={resolution} によるピンポイント周波数補正を開始します (全{len(st_list1_mode)}モード)...")

            refined_freqs = []
            refined_Qs = []
            refined_As = []

            for st1 in range(len(st_list1_mode)):
                coarse_freq = float(st_list1_mode['freq'][st1])
                coarse_Q = float(st_list1_mode['Q'][st1])
                coarse_A = float(st_list1_mode['|A|'][st1])

                excon_parts = st_list1_mode['excon'][st1].split(',')
                sfield_ref = excon_parts[0]
                # excon には実座標 (period * ratio) が入っているため period で割って比率に戻す
                src_pos_ref = [float(val) / period for val in excon_parts[1:4]]

                detcon_parts = st_list1_mode['detcon'][st1].split(',')
                mfield_ref = [detcon_parts[0]]
                m_pos_ref = [[float(val) / period for val in detcon_parts[1:4]], [0.0, 0.0, 0.0]]

                sym_str = st_list1_mode['sym'][st1]
                p_ref = [p_dict[sym_str[0]], p_dict[sym_str[1]], p_dict[sym_str[2]]]

                mc_ref = mc.my_mp_calc(1)
                mc_ref.eps_averaging = True  # 高解像度補正ではサブピクセル平滑化をON
                mc_ref.save_h5 = False
                mc_ref.cutoff_harminv = cutoff_harminv
                mc_ref.progress_tracker = progress

                meep_t_ref = (2.0 * cutoff_harminv / df_refine) + tmax_refine
                label_ref = f"STEP2.5 周波数補正(res={resolution}, 粗周波数={coarse_freq:.4f})"
                progress.start_task(meep_t_ref, resolution, label_ref)

                sim_param_ref = [
                    __file__, resolution, dist_to_pml, sp, pml_w, p_ref,
                    period, n, coarse_freq, df_refine, tmax_refine
                ]
                mc_ref.set_base_param(
                    *sim_param_ref, src_pos_ref, m_pos_ref, mfield_ref, sfield_ref
                )
                apply_structure(mc_ref)
                mc_ref.start_simulation()
                progress.finish_task()
                mp.all_wait()

                # 検出されたモードから元の周波数に最も近い有効モードを選択
                best_f, best_q, best_a = coarse_freq, coarse_Q, coarse_A
                min_diff = 1e9
                for fname in mc_ref.get_file_name():
                    f_list, q_list, a_list, _ = mc_ref.read_harminv_file(fname + '.txt')
                    for f_v, q_v, a_v in zip(f_list, q_list, a_list):
                        diff = abs(f_v - coarse_freq)
                        if q_v > 20 and diff < (df_refine * 0.6) and diff < min_diff:
                            min_diff = diff
                            best_f = f_v
                            best_q = q_v
                            best_a = abs(a_v)

                refined_freqs.append(best_f)
                refined_Qs.append(best_q)
                refined_As.append(best_a)

                if mp.am_master():
                    print(
                        f"[STEP 2.5 補正完了] Mode {st1 + 1}: "
                        f"freq {coarse_freq:.5f} (res={res_harminv}) ➔ {best_f:.5f} (res={resolution}), "
                        f"Q: {coarse_Q:.1f} ➔ {best_q:.1f}"
                    )

            st_list1_mode['freq'] = refined_freqs
            st_list1_mode['Q'] = refined_Qs
            st_list1_mode['|A|'] = refined_As
            if mp.am_master():
                st_list1_mode.to_csv(target_simcon, sep='\t', index=False)
                print(f"[STEP 2.5] 補正後の高解像度共振条件で '{target_simcon}' を更新しました。")
            mp.all_wait()

        # ----------------------------------------------------------------------
        # ★ STEP 3: Mode プロファイル計算 (res=80)
        # ----------------------------------------------------------------------
        flag = 2
        st_list2_mode = [0]

        if len(st_list1_mode) == 0:
            if mp.am_master():
                print(f"[警告] '{target_simcon}' に有効な共振条件が含まれていないため、Mode計算を終了します。")
        else:
            mc_mode_init = mc.my_mp_calc(flag)
            mode_run_date = mc_mode_init.get_date()
            if mp.am_master():
                with open(f'./{mode_run_date}_mode.txt', 'w') as date_f:
                    date_f.write('#date\tstlist1\tst_list2\n')

            mc_mode_list = []
            mode_idx = -1

            for st1 in range(len(st_list1_mode)):
                excon_parts = st_list1_mode['excon'][st1].split(',')
                src_field_m = [excon_parts[0]]
                # excon の実座標を period 比に変換
                src_pos_m = [[float(val) / period for val in excon_parts[1:4]]]
                fcen_m = float(st_list1_mode['freq'][st1])

                sym_str = st_list1_mode['sym'][st1]
                p_m = [p_dict[sym_str[0]], p_dict[sym_str[1]], p_dict[sym_str[2]]]

                for st2 in range(len(st_list2_mode)):
                    mode_idx += 1
                    mc_m = mc.my_mp_calc(flag)
                    mc_m.eps_averaging = True
                    mc_m.save_h5 = SAVE_H5_FILES
                    mc_m.cutoff_mode = cutoff_mode
                    mc_m.progress_tracker = progress
                    mc_mode_list.append(mc_m)
                    sim_date = mc_mode_list[mode_idx].get_date()

                    if mp.am_master():
                        with open(f'./{mode_run_date}_mode.txt', mode='a') as date_f:
                            date_f.write(f"{sim_date}\t{st_list1_mode['file name'][st1]}\t{st_list2_mode[st2]:.3f}\n")

                    for tmax in t_list_mode:
                        for sn in range(len(src_pos_m)):
                            for l in range(len(src_field_m)):
                                meep_t_mode = (2.0 * cutoff_mode / df_mode) + tmax + extra_mode_t
                                label_m = f"STEP3 Mode計算(res={resolution}, f={fcen_m:.5f}, src={src_field_m[l]})"
                                progress.start_task(meep_t_mode, resolution, label_m)

                                sim_param = [__file__, resolution, dist_to_pml, sp, pml_w, p_m, period, n, fcen_m, df_mode, tmax]

                                mc_mode_list[mode_idx].set_base_param(
                                    *sim_param,
                                    src_pos_m[sn],
                                    src_field_m[l],
                                    m_field,
                                    gif_cut_pos,
                                    fcen_m
                                )
                                apply_structure(mc_mode_list[mode_idx])
                                mc_mode_list[mode_idx].start_simulation()
                                progress.finish_task()

