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

    def cluster_and_rank_modes(self, Ath=1e-5, q_min=20.0, f_range=None):
        """
        STEP 2 簡略化アルゴリズム (§5, §6-3):
        - ヒストグラムによる多数決を廃止し、全監視点の結果を閾値 (|A| > Ath, Q > q_min) でフィルタ。
        - 対称性 (sym) ごとに周波数の近さ (|Δf| < f/(2Q)) でクラスタリング・重複除去。
        - 各クラスタの Q の中央値と |A| の最大値を算出し、順位付けして
          `best condition {af_date}.txt` を出力する（列構成は現行互換を維持）。
        """
        if not mp.am_master():
            return pd.DataFrame()

        out_cols = ['sym', 'ex freq', 'file name', 'excon', 'detcon', 'freq', '|A|', 'Q', 'hist n']
        if self.pd_list is None or self.pd_list.empty:
            empty_df = pd.DataFrame(columns=out_cols)
            best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
            empty_df.to_csv(best_path, sep='\t')
            return empty_df

        df_valid = self.pd_list[(self.pd_list['Q'] > q_min) & (self.pd_list['|A|'] > Ath)].copy()
        if f_range is not None and len(f_range) == 2:
            f_min, f_max = float(f_range[0]), float(f_range[1])
            df_valid = df_valid[(df_valid['freq'] >= f_min) & (df_valid['freq'] <= f_max)]

        records = []
        for sym_val in pd.unique(df_valid['sym']):
            sub_df = df_valid[df_valid['sym'] == sym_val].sort_values(by='|A|', ascending=False).reset_index(drop=True)
            used = np.zeros(len(sub_df), dtype=bool)

            for i in range(len(sub_df)):
                if used[i]:
                    continue
                f_seed = float(sub_df.loc[i, 'freq'])
                q_seed = max(float(sub_df.loc[i, 'Q']), q_min)
                half_width_seed = f_seed / (2.0 * q_seed)

                # |Δf| < f/(2Q) に基づく近傍モードのグループ化
                # (超高Qモードでの数値ばらつきによる同一モード分裂を防ぐため最小許容幅 1e-4 を併用)
                cluster_indices = [i]
                used[i] = True
                for j in range(i + 1, len(sub_df)):
                    if used[j]:
                        continue
                    f_cand = float(sub_df.loc[j, 'freq'])
                    q_cand = max(float(sub_df.loc[j, 'Q']), q_min)
                    tol = max(half_width_seed, f_cand / (2.0 * q_cand), 1e-4)
                    if abs(f_cand - f_seed) < tol:
                        cluster_indices.append(j)
                        used[j] = True

                cluster = sub_df.iloc[cluster_indices]
                q_med = float(np.median(cluster['Q'].to_numpy()))
                a_max = float(np.max(cluster['|A|'].to_numpy()))
                f_med = float(np.median(cluster['freq'].to_numpy()))

                # |A| が最大の行を代表行として excon, detcon, file name を保持
                best_row = cluster.iloc[0]
                records.append({
                    'sym': best_row['sym'],
                    'ex freq': best_row['ex freq'],
                    'file name': best_row['file name'],
                    'excon': best_row['excon'],
                    'detcon': best_row['detcon'],
                    'freq': f_med,
                    '|A|': a_max,
                    'Q': q_med,
                    'hist n': int(len(cluster)),
                })

        best_pdlist = pd.DataFrame(records, columns=out_cols)
        if not best_pdlist.empty:
            # Qの中央値と |A| の最大値で順位付け (§6-3)
            best_pdlist.sort_values(by=['Q', '|A|'], ascending=[False, False], inplace=True)
            best_pdlist.reset_index(drop=True, inplace=True)

        best_path = os.path.join(self.directory, f'best condition {self.af_date}.txt')
        best_pdlist.to_csv(best_path, sep='\t')
        return best_pdlist

    def hist_analysis(self, nth=1, max_cons=1, sort_basis=2, bin_num=300):
        """後方互換用のダミーメソッド（§6-3 によりヒストグラム出力は省略）"""
        pass

    def correlated_simco(self, nth=1, Ath=1e-5, bin_num=300, q_min=20.0, f_range=None):
        """後方互換インターフェース: 簡略化されたクラスタリング＆ランキングを実行"""
        return self.cluster_and_rank_modes(Ath=Ath, q_min=q_min, f_range=f_range)

    def start_analysis(self):
        if self.flag == 0:
            return self.analysis_harminv_file(self.pd_list, f"harminv_summary{self.af_date}.txt")
        return pd.DataFrame()
