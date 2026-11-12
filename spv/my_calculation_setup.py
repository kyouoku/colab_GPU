import datetime
import glob
import math
import os
import re
import sys
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import my_mp_setup as mmp


class _DualOutput:
    """ログファイルとコンソール画面 (sys.__stdout__) の両方にリアルタイム出力するストリーム"""
    def __init__(self, log_file):
        self.terminal = sys.__stdout__
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


class my_mp_calc:
    max_flag = 100

    def __init__(self, *args):
        self.flag = self.max_flag
        self.geom_setup_flag = False
        self.sim_setup_flag = False
        self.log_file = None
        self.dt_now = 0
        self.date = ""
        self.fcen = 0
        self.df = 100
        self.tmax = 0
        self.h_file_info = []
        self.mpset = None
        self.p = []
        self.base_param = []
        self.sim_param = []
        self.st_param = []
        self.file_name_list = []

        # 高速化・進捗管理用のオプション属性
        self.eps_averaging = False
        self.save_h5 = False
        self.cutoff_harminv = 4.0
        self.cutoff_mode = 3.0
        self.progress_tracker = None

        if len(args) == 2 and isinstance(args[0], my_mp_calc):
            src_calc = args[0]
            self.flag = src_calc.flag
            self.dt_now = src_calc.dt_now
            self.date = args[1]
            self.fcen = src_calc.fcen
            self.df = src_calc.df
            self.tmax = src_calc.tmax
            self.sim_setup_flag = src_calc.sim_setup_flag
            self.geom_setup_flag = src_calc.geom_setup_flag
            self.eps_averaging = src_calc.eps_averaging
            self.save_h5 = src_calc.save_h5
            self.cutoff_harminv = src_calc.cutoff_harminv
            self.cutoff_mode = src_calc.cutoff_mode
            self.progress_tracker = src_calc.progress_tracker

            self.set_base_param(
                *src_calc.sim_param,
                self.fcen,
                self.df,
                self.tmax,
                *src_calc.base_param
            )
            if self.geom_setup_flag:
                self.set_structure(*src_calc.st_param)

        elif len(args) >= 1 and isinstance(args[0], int):
            self.flag = args[0]
            self.dt_now = datetime.datetime.now()
            self.date = self.dt_now.strftime('%Y%m%d-%H%M%S')

            # マスタープロセスはログファイルとコンソール画面の両方へ出力
            if mp.am_master():
                if sys.stdout is not sys.__stdout__ and hasattr(sys.stdout, 'close'):
                    try:
                        sys.stdout.close()
                    except Exception:
                        pass
                raw_log = open(f"{self.date}_log.out", 'w', buffering=1)
                self.log_file = _DualOutput(raw_log)
                sys.stdout = self.log_file
                print(f'my_calculation_setup.init: flag = {self.flag}')
        else:
            print('my_mp_calc Error: initialization is failed')

    def read_harminv_file(self, harminv_fname):
        f, Q, amp, Er = [], [], [], []
        if not os.path.exists(harminv_fname):
            print(f"Warning: {harminv_fname} does not exist.")
            return f, Q, amp, Er

        with open(harminv_fname, "r") as test_data:
            lines = test_data.readlines()

        for i in range(1, len(lines)):
            temp = lines[i].split('\t')
            if len(temp) > 4:
                f.append(float(temp[0]))
                Q.append(float(temp[2]))
                amp.append(complex(temp[3].strip()))
                Er.append(complex(temp[4].strip()))
            else:
                break

        # ★ 分類A-⑤: 1行ごとのprintによるI/Oフラッシュ待ちを避け、ファイル単位のサマリー出力に集約
        print(f"read\t: {harminv_fname} ({len(f)} modes)")
        return f, Q, amp, Er

    def set_base_param(self, py_name, resolution, dist_to_pml, sp, pml_w, p, period, n, fcen, df, tmax, *args):
        print(f'my_calculation_setup.set_base_param: flag = {self.flag}')
        self.p = p
        self.fcen = fcen
        self.df = df
        self.tmax = tmax

        pml_str = ' '.join([f'{num:.2f}' for num in dist_to_pml])
        self.mpset = mmp.my_mp_setup(
            py_name, f"[{pml_str}]{self.date}", resolution, *sp, pml_w, *p, period, n
        )
        # 高速化・進捗管理用の属性を my_mp_setup に引き渡し
        self.mpset.eps_averaging = self.eps_averaging
        self.mpset.save_h5 = self.save_h5
        self.mpset.cutoff_harminv = self.cutoff_harminv
        self.mpset.cutoff_mode = self.cutoff_mode
        self.mpset.progress_tracker = self.progress_tracker

        self.sim_setup_flag = True
        self.sim_param = [py_name, resolution, dist_to_pml, sp, pml_w, p, period, n]

        if self.flag == 0:
            if len(args) == 0:
                self.file_name_list = self.mpset.get_spectral_fname(self.fcen, self.df, self.tmax)
            else:
                self.flag = self.max_flag

        elif self.flag == 1:
            if len(args) == 4 and len(args[1][0]) == 3:
                self.src_pos = args[0]
                self.m_pos = args[1]
                self.mfield_list = args[2]
                self.sfield = args[3]
                self.file_name_list = self.mpset.get_harminv_fname(
                    self.src_pos, self.m_pos, self.fcen, self.df, self.tmax, self.mfield_list, self.sfield
                )
            else:
                print("Error my_calculation_setup flag 1: the number of args is not appropriate")
                self.flag = self.max_flag

        elif self.flag in (2, 2.1):
            if len(args) == 5 and isinstance(args[4], list) and len(args[4]) == 6:
                self.src_pos = args[0]
                self.sfield = args[1]
                self.out_field_list = args[2]
                self.pos_xyz = args[3]
                self.h_file_info = args[4]
                harminv_fname = self.mpset.get_harminv_fname(
                    self.src_pos, self.h_file_info[1], self.fcen,
                    self.h_file_info[2], self.h_file_info[3], self.h_file_info[4], self.h_file_info[5]
                )
                name_list = []
                for hfname in harminv_fname:
                    target_file = re.sub(r'[0-9]{8}-[0-9]{6}', self.h_file_info[0], hfname) + '.txt'
                    f_list, _, _, _ = self.read_harminv_file(target_file)
                    for freq_val in f_list:
                        name_list.append(
                            self.mpset.get_mode_fname(
                                self.src_pos, self.sfield, freq_val, self.df, self.tmax, self.pos_xyz, self.out_field_list
                            )
                        )
                self.file_name_list = name_list
                self.flag = 2
            elif len(args) == 5:
                self.src_pos = args[0]
                self.sfield = args[1]
                self.out_field_list = args[2]
                self.pos_xyz = args[3]
                self.fcen = args[4]
                self.file_name_list = self.mpset.get_mode_fname(
                    self.src_pos, self.sfield, self.fcen, self.df, self.tmax, self.pos_xyz, self.out_field_list
                )
                self.flag = 2.1
            else:
                self.flag = self.max_flag

        elif self.flag == 3:
            if len(args) == 4 and len(args[3]) == 6:
                self.src_pos = args[0]
                self.md_ex_df = args[1]
                self.out_field_list = args[2]
                self.h_file_info = args[3]
                harminv_fname = self.mpset.get_harminv_fname(
                    self.src_pos, self.h_file_info[1], self.fcen,
                    self.h_file_info[2], self.h_file_info[3], self.h_file_info[4], self.h_file_info[5]
                )
                name_list = []
                for hfname in harminv_fname:
                    target_file = re.sub(r'[0-9]{8}-[0-9]{6}', self.h_file_info[0], hfname) + '.txt'
                    f_list, _, _, _ = self.read_harminv_file(target_file)
                    for freq_val in f_list:
                        name_list.append(
                            self.mpset.get_freq_solver_fname(self.src_pos, self.df, freq_val, self.out_field_list)
                        )
                self.file_name_list = name_list
            else:
                self.flag = self.max_flag

        elif self.flag == 4:
            self.file_name_list = 'test'
        elif self.flag == 5:
            self.file_name_list = 'file_removal'
        else:
            self.flag = self.max_flag
            self.file_name_list = ''

        self.base_param = list(args)

    def set_structure(self, *args):
        if len(args) > 1:
            self.mpset.set_structure(*args)
            self.geom_setup_flag = True
            if self.sim_setup_flag and self.flag == 1:
                self.file_name_list = self.mpset.get_harminv_fname(
                    self.src_pos,
                    self.m_pos,
                    self.fcen,
                    self.df,
                    self.tmax,
                    self.mfield_list,
                    self.sfield,
                )
        else:
            print('Error: set_structure needs two arguments at least')
        self.st_param = list(args)

    def get_file_name(self):
        return self.file_name_list

    def start_simulation(self):
        def line(n):
            for _ in range(n):
                print("*" * 118)

        if self.sim_setup_flag and self.geom_setup_flag:
            self.mpset.check_epsilon()

            if self.flag == 0:
                line(1)
                self.mpset.spectral_calc(self.fcen, self.df, self.tmax)
                line(1)
            elif self.flag == 1:
                line(1)
                self.mpset.harminv_calc(
                    self.src_pos, self.m_pos, self.fcen, self.df, self.tmax, self.mfield_list, self.sfield
                )
                line(1)
            elif self.flag == 2:
                harminv_fname = self.mpset.get_harminv_fname(
                    self.src_pos, self.h_file_info[1], self.fcen,
                    self.h_file_info[2], self.h_file_info[3], self.h_file_info[4], self.h_file_info[5]
                )
                for hfname in harminv_fname:
                    target_file = re.sub(r'[0-9]{8}-[0-9]{6}', self.h_file_info[0], hfname) + '.txt'
                    f_list, Q_list, A_list, Er_list = self.read_harminv_file(target_file)
                    if len(f_list) != 0:
                        line(2)
                        for l in range(len(f_list)):
                            line(1)
                            print(f"fcen = : {f_list[l]:.3f}, Q : {Q_list[l]}, amp : {abs(A_list[l])}, error : {abs(Er_list[l])}")
                            self.mpset.mode_calc(
                                self.src_pos, self.sfield, f_list[l], self.df, self.tmax, self.pos_xyz, self.out_field_list
                            )
                            line(1)
                        line(2)
            elif self.flag == 2.1:
                self.mpset.mode_calc(
                    self.src_pos, self.sfield, self.fcen, self.df, self.tmax, self.pos_xyz, self.out_field_list
                )
                self.flag = 2
            elif self.flag == 3:
                harminv_fname = self.mpset.get_harminv_fname(
                    self.src_pos, self.h_file_info[1], self.fcen,
                    self.h_file_info[2], self.h_file_info[3], self.h_file_info[4], self.h_file_info[5]
                )
                for hfname in harminv_fname:
                    target_file = re.sub(r'[0-9]{8}-[0-9]{6}', self.h_file_info[0], hfname) + '.txt'
                    f_list, Q_list, A_list, Er_list = self.read_harminv_file(target_file)
                    if len(f_list) != 0:
                        line(2)
                        for l in range(len(f_list)):
                            line(1)
                            self.mpset.freq_solver(self.src_pos, self.df, f_list[l], self.out_field_list)
                            line(1)
                        line(2)
            elif self.flag == 5:
                self.mpset.remove_glob('./_t*_hz.png')
            else:
                self.flag = self.max_flag
                self.file_name_list = ''
        elif not self.sim_setup_flag:
            print('simulation setup is not initialized.')
        else:
            print('geometry is not initialized.')

    def get_sim_condition(self):
        sim_condition = []
        for fname in self.file_name_list:
            temp = fname.split('_')
            temp0 = temp[0].split(')')
            ex = temp0[1]
            ex_pos = temp0[0].split('(')[-1]
            ex_x, ex_y, ex_z = ex_pos.split(' ')
            sym = temp[1].split(' ')[-1].replace(')', '')
            m_ex, m_pos = temp[3].replace(')', '').split('(')
            m_x, m_y, m_z = m_pos.split(' ')
            freq = temp[0].split('(')[1].split(' ')[0]
            sim_condition.append(
                [sym, freq, ex, ex_x, ex_y, ex_z, m_ex, m_x, m_y, m_z]
            )
        return sim_condition

    def get_sim_condition_index(self):
        return ['sym', 'ex freq', 'ex field', 'ex X', 'ex Y', 'ex Z', 'det field', 'det X', 'det Y', 'det Z']

    def get_date(self):
        return self.date

