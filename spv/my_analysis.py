import glob
import math
import os
import re
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd


class my_analysis:
    max_flag = 100

    def __init__(self, flag):
        self.flag = flag
        self.freq = []
        self.Q = []
        self.A = []
        self.Er = []
        self.pd_list = None
        self.af_date = ""
        self.directory = ""

    def set_base_param(self, *args):
        if self.flag == 0:
            if len(args) != 3:
                print("directory")
                print("file name, list")
                print("output filename, string")
                print("comment, string list")
                self.flag = self.max_flag
            else:
                self.directory = os.path.join(args[0], '')
                self.pd_list = args[1]
                self.af_date = args[2]
        else:
            print("flag should be within 0")
            print("0: analysis_harminv_file")
            print("others: none")

    def analysis_harminv_file(self, pd_list, output_fname):
        def read_harminv_file(harminv_fname):
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
            return f, Q, amp, Er

        add_columns = ['freq', 'Q', 'Re[A]', 'Im[A]', '|A|', 'Re[Er]']
        all_columns = list(pd_list.columns) + add_columns

        # DataFrame.locの多重ループ参照を排除し、タプル展開で高速化
        rows = []
        base_records = pd_list.to_numpy()
        fname_idx = pd_list.columns.get_loc('file name')

        for row_vals in base_records:
            harminv_path = f"{self.directory}{row_vals[fname_idx]}.txt"
            f_vals, Q_vals, A_vals, Er_vals = read_harminv_file(harminv_path)
            base_list = list(row_vals)
            for f_v, q_v, a_v, er_v in zip(f_vals, Q_vals, A_vals, Er_vals):
                rows.append(base_list + [f_v, q_v, a_v.real, a_v.imag, abs(a_v), er_v.real])

        newpd_list = pd.DataFrame(rows, columns=all_columns)

        newpd_list['excon'] = (
            newpd_list['ex field'].astype(str) + ',' +
            newpd_list['ex X'].astype(str) + ',' +
            newpd_list['ex Y'].astype(str) + ',' +
            newpd_list['ex Z'].astype(str)
        )
        newpd_list['detcon'] = (
            newpd_list['det field'].astype(str) + ',' +
            newpd_list['det X'].astype(str) + ',' +
            newpd_list['det Y'].astype(str) + ',' +
            newpd_list['det Z'].astype(str)
        )
        newpd_list.drop(
            columns=['ex field', 'ex X', 'ex Y', 'ex Z', 'det field', 'det X', 'det Y', 'det Z'],
            inplace=True
        )

        self.pd_list = newpd_list
        if mp.am_master():
            newpd_list.to_csv(output_fname, sep='\t')
        return newpd_list

    def hist_analysis(self, nth, max_cons, sort_basis, bin_num):
        if not mp.am_master():
            return

        def graph_output(x_bins, y_vals, bins, n, fname, y_label):
            np.savetxt(f"{fname}.txt", np.column_stack((x_bins, y_vals)), fmt="%s", delimiter="\t", newline="\r\n")

            fig, ax1 = plt.subplots()
            ax1.set_yscale('log')
            ax1.plot(bins[:-1], n, '-r', lw=0.2, alpha=0.6)
            ax1.set_ylabel("Harminv-captured freqs. histgram (log)")
            ax1.set_xlabel("frequency")

            ax2 = ax1.twinx()
            ax2.set_yscale('log')
            ax2.scatter(x_bins, y_vals, s=0.1, c="blue", marker='.', alpha=0.8)
            ax2.set_ylabel("log " + y_label)
            plt.title(y_label + " histgram", fontsize=15)
            # ★ 分類A-⑤: dpi=600 ➔ dpi=200 に変更し画像レンダリング・保存時間を短縮
            fig.savefig(f"{fname}.png", dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        def histgram_analysis(p_str, bins, n, hist_f, hist_A, hist_Q, hist_Er):
            np.savetxt(f"f-hist{p_str} {self.af_date}.txt", np.column_stack((bins[:-1], n)), fmt="%s", delimiter="\t", newline="\r\n")

            # 2重ループを np.searchsorted でベクトル化し高速化
            bin_indices = np.searchsorted(bins, hist_f, side='right') - 1
            bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
            mapped_bins = bins[bin_indices]

            graph_output(mapped_bins, hist_Er, bins, n, f'f-dist Re[Er]{p_str} {self.af_date}', 'Re[Er]')
            graph_output(mapped_bins, hist_A, bins, n, f'f-dist abs(A){p_str} {self.af_date}', '|A|')
            graph_output(mapped_bins, np.abs(hist_Q), bins, n, f'f-dist abs(Q){p_str} {self.af_date}', '|Q|')

        unique_sym = pd.unique(self.pd_list['sym'])
        for sym_val in unique_sym:
            sub_df = self.pd_list[self.pd_list['sym'] == sym_val]
            hist_f = sub_df['freq'].to_numpy()

            fig, ax = plt.subplots(facecolor="w")
            n, bins, _ = ax.hist(hist_f, bins=bin_num)
            fig.savefig(f'f-hist{sym_val} {self.af_date}.png', dpi=200)
            plt.close(fig)

            hist_Q = sub_df['Q'].to_numpy()
            hist_A = sub_df['|A|'].to_numpy()
            hist_Er = sub_df['Re[Er]'].to_numpy()
            histgram_analysis(sym_val, bins, n, hist_f, hist_A, hist_Q, hist_Er)

    def correlated_simco(self, nth, Ath, bin_num):
        if not mp.am_master():
            return

        unique_sym = pd.unique(self.pd_list['sym'])
        hn, hb = [], []

        sym_list, exf_list, fname_list = [], [], []
        excon_list, detcon_list, freq_list = [], [], []
        A_list, Q_list, hist_n_list = [], [], []

        # Figureを生成せず np.histogram で直接高速計算
        for sym_val in unique_sym:
            hist_f = self.pd_list[self.pd_list['sym'] == sym_val]['freq'].to_numpy()
            n, bins = np.histogram(hist_f, bins=bin_num)
            hn.append(n)
            hb.append(bins)

        for i, sym_val in enumerate(unique_sym):
            hn_nd = hn[i]
            hb_nd = hb[i][:len(hn_nd)]
            ind = np.where(hn_nd > nth)[0]

            hist_f = self.pd_list[self.pd_list['sym'] == sym_val]

            for idx in ind:
                if idx + 1 < len(hb_nd):
                    new_hf = hist_f[(hist_f['freq'] >= hb_nd[idx]) & (hist_f['freq'] <= hb_nd[idx + 1])]
                    freq_str = f'{(hb_nd[idx] + hb_nd[idx + 1]) / 2:.4f}'
                else:
                    new_hf = hist_f[hist_f['freq'] >= hb_nd[idx]]
                    freq_str = f'{hb_nd[idx]:.4f}'

                new_hf2 = new_hf[new_hf['|A|'] > Ath]

                if not new_hf2.empty:
                    ga1 = new_hf2[['|A|', 'detcon']].groupby('detcon').median().sort_values(by='|A|', ascending=False)
                    ga_len = min(len(ga1.index), 30)

                    fig, ax = plt.subplots()
                    plt.xticks(rotation=90)
                    ax.grid(which="major", axis="x", color="gray", alpha=0.2, linestyle="--", linewidth=0.2)
                    ax.scatter(ga1.index[:ga_len], ga1['|A|'].iloc[:ga_len], s=1, c="red", marker='.', alpha=0.8)
                    base_det_name = f'A sort detcon median  f{freq_str}{sym_val} {self.af_date}'
                    fig.savefig(f'{base_det_name}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    ga1.to_csv(f'{base_det_name}.txt', sep='\t')

                    ga2 = new_hf2[['|A|', 'excon']].groupby('excon').median().sort_values(by='|A|', ascending=False)
                    ga_len = min(len(ga2.index), 30)

                    fig, ax = plt.subplots()
                    plt.xticks(rotation=90)
                    ax.grid(which="major", axis="x", color="gray", alpha=0.2, linestyle="--", linewidth=0.2)
                    ax.scatter(ga2.index[:ga_len], ga2['|A|'].iloc[:ga_len], s=1, c="red", marker='.', alpha=0.8)
                    base_ex_name = f'A sort excon median  f{freq_str}{sym_val} {self.af_date}'
                    fig.savefig(f'{base_ex_name}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    ga2.to_csv(f'{base_ex_name}.txt', sep='\t')

                    best = pd.DataFrame()
                    top_excon = ga2.index[0]
                    for det_k in ga1.index:
                        best = new_hf2[(new_hf2['excon'] == top_excon) & (new_hf2['detcon'] == det_k)]
                        if len(best) != 0:
                            break

                    if not best.empty:
                        best_row = best.iloc[0]
                        sym_list.append(best_row['sym'])
                        exf_list.append(best_row['ex freq'])
                        fname_list.append(best_row['file name'])
                        excon_list.append(best_row['excon'])
                        detcon_list.append(best_row['detcon'])
                        freq_list.append(best_row['freq'])
                        A_list.append(best_row['|A|'])
                        Q_list.append(best_row['Q'])
                        hist_n_list.append(new_hf2.size)

        best_pdlist = pd.DataFrame({
            'sym': sym_list,
            'ex freq': exf_list,
            'file name': fname_list,
            'excon': excon_list,
            'detcon': detcon_list,
            'freq': freq_list,
            '|A|': A_list,
            'Q': Q_list,
            'hist n': hist_n_list
        })
        best_pdlist.sort_values(by='freq', inplace=True, ascending=True)
        best_pdlist.to_csv(f'best condition {self.af_date}.txt', sep='\t')

    def start_analysis(self):
        if self.flag == 0:
            return self.analysis_harminv_file(self.pd_list, f"harminv_summary{self.af_date}.txt")

