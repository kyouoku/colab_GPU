import datetime
import os
import re
import sys
import meep as mp
import numpy as np

try:
    from . import my_mp_setup as mmp
except ImportError:
    import my_mp_setup as mmp


class _DualOutput:
    """コンソール画面 (sys.__stdout__)、run_log.txt、および個別ログファイルへリアルタイム出力するストリーム"""

    def __init__(self, log_file, run_log_file=None):
        self.terminal = sys.__stdout__
        self.log_file = log_file
        self.run_log_file = run_log_file

    def write(self, message):
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
        if self.run_log_file and not self.run_log_file.closed:
            self.run_log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
        if self.run_log_file and not self.run_log_file.closed:
            self.run_log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        if self.run_log_file and not self.run_log_file.closed:
            self.run_log_file.close()


class my_mp_calc:
    max_flag = 100
    run_log_path = "run_log.txt"

    def __init__(self, *args):
        self.flag = self.max_flag
        self.sim_setup_flag = False
        self.mpset = None
        self.sim_param = []
        self.file_name_list = []

        self.eps_averaging = False
        self.save_h5 = False
        self.cutoff_harminv = 5.0
        self.cutoff_mode = 5.0
        self.decay_dt = 50
        self.decay_by = 1e-5
        self.use_decay_stop = True
        self.src_seed = 42
        self.shot_id = None
        self.capture_dft = False
        self.dft_nfreq = 11
        self.decay_pt = None
        self.progress_tracker = None

        if len(args) == 2 and isinstance(args[0], my_mp_calc):
            src_calc = args[0]
            self.flag = args[1]
            self.dt_now = src_calc.dt_now
            self.date = src_calc.date

            self.eps_averaging = src_calc.eps_averaging
            self.save_h5 = src_calc.save_h5
            self.cutoff_harminv = src_calc.cutoff_harminv
            self.cutoff_mode = src_calc.cutoff_mode
            self.decay_dt = src_calc.decay_dt
            self.decay_by = src_calc.decay_by
            self.use_decay_stop = src_calc.use_decay_stop
            self.src_seed = src_calc.src_seed
            self.shot_id = src_calc.shot_id
            self.capture_dft = src_calc.capture_dft
            self.dft_nfreq = src_calc.dft_nfreq
            self.decay_pt = src_calc.decay_pt
            self.progress_tracker = src_calc.progress_tracker

            self.set_base_param(
                *src_calc.sim_param,
                src_calc.fcen,
                src_calc.df,
                src_calc.tmax,
                src_calc.src_pos,
                src_calc.sfield,
                src_calc.out_field_list,
                src_calc.pos_xyz,
                src_calc.h_file_info
            )
            self.set_structure(*src_calc.st_param)
        elif len(args) == 1 and isinstance(args[0], (int, float)):
            self.flag = args[0]
            self.dt_now = datetime.datetime.now()
            self.date = self.dt_now.strftime('%Y%m%d-%H%M%S')

            if mp.am_master():
                if sys.stdout is not sys.__stdout__ and hasattr(sys.stdout, 'close'):
                    try:
                        sys.stdout.close()
                    except Exception:
                        pass
                raw_log = open(f"{self.date}_log.out", 'w', buffering=1)
                run_log = open(self.run_log_path, 'a', buffering=1)
                self.log_file = _DualOutput(raw_log, run_log)
                sys.stdout = self.log_file
                print(f'my_calculation_setup.init: flag = {self.flag}')
        else:
            print('my_calculation_setup.init: input argument error')
            self.flag = self.max_flag

    def read_harminv_file(self, harminv_fname):
        f, Q, amp, Er = [], [], [], []
        if not os.path.exists(harminv_fname):
            return f, Q, amp, Er
        with open(harminv_fname, "r") as test_data:
            lines = test_data.readlines()
        for line in lines[1:]:
            temp = line.split('\t')
            if len(temp) > 4:
                f.append(float(temp[0]))
                Q.append(float(temp[2]))
                amp.append(complex(temp[3].strip()))
                Er.append(complex(temp[4].strip()))
            else:
                break

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
        self.mpset.eps_averaging = self.eps_averaging
        self.mpset.save_h5 = self.save_h5
        self.mpset.cutoff_harminv = self.cutoff_harminv
        self.mpset.cutoff_mode = self.cutoff_mode
        self.mpset.decay_dt = self.decay_dt
        self.mpset.decay_by = self.decay_by
        self.mpset.use_decay_stop = self.use_decay_stop
        self.mpset.src_seed = self.src_seed
        self.mpset.shot_id = self.shot_id
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
        else:
            self.flag = self.max_flag

    def set_structure(self, *args):
        if self.sim_setup_flag:
            self.st_param = list(args)
            self.mpset.set_structure(*args)
        else:
            print('set_base_param first')

    def start_simulation(self):
        def line(n=1):
            print('#' * 100 * n)

        if self.sim_setup_flag:
            self.mpset.check_epsilon()
            if self.flag == 0:
                line(1)
                self.mpset.spectral_calc(self.fcen, self.df, self.tmax)
                line(1)
            elif self.flag == 1:
                line(1)
                self.mpset.harminv_calc(
                    self.src_pos, self.m_pos, self.fcen, self.df, self.tmax,
                    self.mfield_list, self.sfield,
                    capture_dft=self.capture_dft, dft_nfreq=self.dft_nfreq
                )
                line(1)
            elif self.flag == 2.1:
                self.mpset.mode_calc(
                    self.src_pos, self.sfield, self.fcen, self.df, self.tmax,
                    self.pos_xyz, self.out_field_list, decay_pt=self.decay_pt
                )
                self.flag = 2
            else:
                print(f'simulation flag is not appropriate: {self.flag}')
        else:
            print('setup simulation first')

    def get_file_name(self):
        return self.file_name_list

    def get_sim_condition(self):
        sim_condition = []
        is_multi_sfield = isinstance(self.sfield, (list, tuple, np.ndarray)) and len(self.sfield) > 1
        for fname in self.file_name_list:
            temp = fname.split('_')
            temp0 = temp[0].split(')')
            ex_pos_str = temp0[1].replace('(', '')
            ex_x, ex_y, ex_z = ex_pos_str.split(' ')
            ex = temp0[2]
            sym = temp[1].split(' ')[-1].replace(')', '')
            m_ex, m_pos = temp[3].replace(')', '').split('(')
            m_x, m_y, m_z = m_pos.split(' ')
            freq = temp[0].split('(')[1].split(' ')[0]
            if is_multi_sfield:
                ex = m_ex
            sim_condition.append(
                [sym, freq, ex, ex_x, ex_y, ex_z, m_ex, m_x, m_y, m_z]
            )
        return sim_condition

    def get_sim_condition_index(self):
        return ['sym', 'ex freq', 'ex field', 'ex X', 'ex Y', 'ex Z', 'det field', 'det X', 'det Y', 'det Z']

    def get_date(self):
        return self.date
