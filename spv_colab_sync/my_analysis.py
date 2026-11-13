import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd


class my_analysis:
    max_flag = 100

    def __init__(self, flag=0):
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
                self.flag = self.max_flag
            else:
                self.directory = os.path.join(args[0], '')
                self.pd_list = args[1]
                self.af_date = args[2]
        else:
            print("flag should be within 0")

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

        if not newpd_list.empty:
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
        else:
            newpd_list = pd.DataFrame(
                columns=['sym', 'ex freq', 'file name', 'freq', 'Q', 'Re[A]', 'Im[A]', '|A|', 'Re[Er]', 'excon', 'detcon']
            )

        self.pd_list = newpd_list
        if mp.am_master():
            out_path = os.path.join(self.directory, output_fname) if self.directory else output_fname
            newpd_list.to_csv(out_path, sep='\t')
        return newpd_list

    def cluster_and_rank_modes(self, Ath=1e-5, q_min=20.0, f_range=None, min_detections=2, rel_freq_tol_floor=2e-3):
        """
        spv_gpu/analysis/select_modes.py と同一のクラスタリング・ランキング基準:
        - |Δf| <= max(f / (2*Q), f * rel_freq_tol_floor) で近接ピークを統合
        - 代表行はクラスタ内の |A| * sqrt(max(Q, 1)) 最大行
        - freq, Q はクラスタ内の中央値、|A| は最大値、Q_max / Q_iqr / n_detections も算出
        - 複数点検出 (n_detections >= min_detections) を優先し、
          rank_score = Q * log10(1 + 10 * |A| / max|A|) で順位付け
        """
        if not mp.am_master():
            return pd.DataFrame()

        out_cols = [
            'sym', 'pz', 'ex freq', 'file name', 'excon', 'detcon',
            'freq', '|A|', 'Re[A]', 'Im[A]', 'Re[Er]',
            'Q', 'Q_max', 'Q_iqr', 'n_detections', 'hist n', 'rank_score',
        ]
        if self.pd_list is None or self.pd_list.empty:
            empty_df = pd.DataFrame(columns=out_cols)
            best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
            empty_df.to_csv(best_path, sep='\t')
            return empty_df

        df_valid = self.pd_list[(self.pd_list['Q'] > q_min) & (self.pd_list['|A|'] > Ath)].copy()
        if f_range is not None and len(f_range) == 2:
            f_min, f_max = float(f_range[0]), float(f_range[1])
            df_valid = df_valid[(df_valid['freq'] >= f_min) & (df_valid['freq'] <= f_max)]

        if df_valid.empty:
            empty_df = pd.DataFrame(columns=out_cols)
            best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
            empty_df.to_csv(best_path, sep='\t')
            return empty_df

        pz_map = {'+': 1, '-': -1, 'o': 0}
        records = []

        for sym_val in pd.unique(df_valid['sym']):
            grp_sorted = df_valid[df_valid['sym'] == sym_val].sort_values(by='freq', ascending=True).reset_index(drop=True)
            used = np.zeros(len(grp_sorted), dtype=bool)
            pz_val = int(pz_map.get(str(sym_val)[-1], 0))

            for i in range(len(grp_sorted)):
                if used[i]:
                    continue
                f_seed = float(grp_sorted.loc[i, 'freq'])
                q_seed = max(float(grp_sorted.loc[i, 'Q']), q_min)
                tol = max(f_seed / (2.0 * q_seed), f_seed * rel_freq_tol_floor)

                diffs = np.abs(grp_sorted['freq'].to_numpy(dtype=np.float64) - f_seed)
                members_mask = (~used) & (diffs <= tol)
                used[members_mask] = True
                members = grp_sorted[members_mask]

                scores = members['|A|'].to_numpy(dtype=np.float64) * np.sqrt(
                    np.maximum(members['Q'].to_numpy(dtype=np.float64), 1.0)
                )
                best_local_idx = int(np.argmax(scores))
                rep_row = members.iloc[best_local_idx]

                f_vals = members['freq'].to_numpy(dtype=np.float64)
                q_vals = members['Q'].to_numpy(dtype=np.float64)
                a_vals = members['|A|'].to_numpy(dtype=np.float64)
                q75, q25 = np.percentile(q_vals, [75, 25]) if len(q_vals) > 1 else (q_vals[0], q_vals[0])

                records.append({
                    'sym': str(rep_row['sym']),
                    'pz': pz_val,
                    'ex freq': rep_row['ex freq'],
                    'file name': str(rep_row['file name']),
                    'excon': str(rep_row['excon']),
                    'detcon': str(rep_row['detcon']),
                    'freq': float(np.median(f_vals)),
                    '|A|': float(np.max(a_vals)),
                    'Re[A]': float(rep_row.get('Re[A]', rep_row['|A|'])),
                    'Im[A]': float(rep_row.get('Im[A]', 0.0)),
                    'Re[Er]': float(np.min(members['Re[Er]'].to_numpy(dtype=np.float64))),
                    'Q': float(np.median(q_vals)),
                    'Q_max': float(np.max(q_vals)),
                    'Q_iqr': float(q75 - q25),
                    'n_detections': int(len(members)),
                    'hist n': int(len(members)),
                })

        clusters_df = pd.DataFrame(records)
        if clusters_df.empty:
            empty_df = pd.DataFrame(columns=out_cols)
            best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
            empty_df.to_csv(best_path, sep='\t')
            return empty_df

        multi_det = clusters_df[clusters_df['n_detections'] >= min_detections]
        pool_df = multi_det.copy() if len(multi_det) >= 1 else clusters_df.copy()

        max_a_global = max(float(pool_df['|A|'].max()), 1e-30)
        pool_df['rank_score'] = pool_df['Q'] * np.log10(1.0 + 10.0 * (pool_df['|A|'] / max_a_global))
        pool_df = pool_df.sort_values(by=['rank_score', 'Q', '|A|'], ascending=[False, False, False]).reset_index(drop=True)

        best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
        pool_df[out_cols].to_csv(best_path, sep='\t')
        return pool_df[out_cols]

    def plot_harminv_summary(self, selected_df=None, out_png_path='harminv_summary_coarse.png', f_range=None):
        """spv_gpu 互換の Harminv 検出分布 (freq vs Q 散布図・ヒストグラム) を保存する。"""
        if not mp.am_master() or self.pd_list is None or self.pd_list.empty:
            return None

        df_plot = self.pd_list[(self.pd_list['Q'] > 10.0) & (self.pd_list['|A|'] > 1e-8)].copy()
        if f_range is not None and len(f_range) == 2:
            df_plot = df_plot[(df_plot['freq'] >= float(f_range[0])) & (df_plot['freq'] <= float(f_range[1]))]
        if df_plot.empty:
            return None

        fig, (ax_q, ax_hist) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=140)
        for sym_label, grp in df_plot.groupby('sym'):
            ax_q.scatter(
                grp['freq'],
                grp['Q'],
                s=np.clip(20.0 + 15.0 * np.log10(np.maximum(grp['|A|'], 1e-12) + 1e-12), 12, 120),
                alpha=0.65,
                label=f'sym={sym_label}',
            )

        if selected_df is not None and not selected_df.empty:
            ax_q.scatter(
                selected_df['freq'],
                selected_df['Q'],
                s=160,
                facecolors='none',
                edgecolors='red',
                linewidths=2.0,
                label='Selected modes',
            )
            for idx, r in selected_df.iterrows():
                ax_q.annotate(
                    f"#{idx + 1} f={r['freq']:.4f}, Q={r['Q']:.0f}",
                    (r['freq'], r['Q']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='red',
                )

        ax_q.set_yscale('log')
        ax_q.set_ylabel('Q factor (log)')
        ax_q.set_title('Harminv Detected Peaks (MPI PyMeep)')
        ax_q.grid(True, which='both', linestyle='--', alpha=0.35)
        ax_q.legend(loc='best', fontsize=8)

        ax_hist.hist(df_plot['freq'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax_hist.set_xlabel('Normalized Frequency (a/lambda or um^-1)')
        ax_hist.set_ylabel('Detection Count')
        ax_hist.grid(True, linestyle='--', alpha=0.35)

        fig.tight_layout()
        full_path = os.path.join(self.directory, out_png_path) if self.directory else out_png_path
        fig.savefig(full_path, bbox_inches='tight')
        plt.close(fig)
        return full_path

    def start_analysis(self):
        if self.flag == 0:
            return self.analysis_harminv_file(self.pd_list, f"harminv_summary{self.af_date}.txt")
        return pd.DataFrame()
