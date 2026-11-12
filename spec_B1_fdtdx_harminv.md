# 仕様書B1: FDTDX（JAX/GPU）＋ 本家Harminv CLI への移植

## 1. 目的・範囲
- Meep FDTD を **FDTDX（JAX）** に置き換え、Colab の **GPU（G4 / A100、次点で L4・T4）** で実行する。TPU v6e/v5e での実行は将来の拡張とする。
- 共振モードの解析には **本家 Harminv の CLI**（libharminv）を使う。
- 対象構造は type 3〜6 のすべて。z対称は偶と奇の両方を扱う。
- パイプライン（STEP 1 → 2 → 2.5 → 3）と出力形式は仕様書Aと共通にする。

## 2. 実行環境
| 項目 | 仕様 |
|---|---|
| ランタイム | G4 / A100（第1候補）、L4 / T4 は小さい構造または draft 用 |
| 導入 | `pip install fdtdx "jax[cuda12]"`、`apt-get install -y harminv`（使えなければ `conda-forge::harminv` を使う） |
| 精度 | float32（既定）。高Qモードの補正だけ float64 を選べるようにする（`x64=True`） |
| 起動時のチェック | `jax.devices()`、GPUメモリ、`harminv --version` をログに出す |

## 3. 構成
```
spv_gpu/
  colab_run.ipynb
  config.py            # ユーザー入力と自動導出（仕様書Aの§4・§5と同じ）
  geometry/
    shapes.py          # 円柱・多角形・ボウタイなどをSDFで描くプリミティブ
    structures.py      # type 3〜6 → 誘電率配列 ε(x,y,z)（my_geometry.py から移植）
    smoothing.py       # サブピクセル平均（体積分率による平均）
  solver/
    fdtdx_runner.py    # セル・PML・波源・検出器の組み立てと実行
    symmetry.py        # z鏡映の実装（§6）
  analysis/
    harminv_cli.py     # 時系列 → harminv 実行 → DataFrame
    select_modes.py    # STEP 2（モード統合とランキング）
  pipeline.py          # STEP 1 → 3
  validate_vs_meep.py  # 現行Meepの結果と比較
```

## 4. ユーザー入力
仕様書Aの§4と同じ（`structure_type`, `structure`, `f_range`, `parities`, `accuracy`（最小特徴寸法あたりの格子数）, `n_modes`, `run_mode`）。以下だけ追加する。
```python
backend = dict(dtype='float32', max_gpu_mem_frac=0.9)
```

## 5. 自動導出
仕様書Aの§5と同じ規則を使う。FDTDX特有の項目は次のとおり。
| 項目 | 規則 |
|---|---|
| 格子 | 一様格子で `dx = period/res`。Courant数は `0.99/√3`（FDTDXの既定値に合わせる） |
| メモリ見積もり | 配列あたり `Nx·Ny·Nz × 4B`。場6成分 ＋ 誘電率 ＋ CPML補助変数で約16配列。GPUメモリを超えそうなら、`res` を下げるよう警告して停止する |
| PML | FDTDX の `PerfectlyMatchedLayer`。厚さは `pml_w`（格子数換算で10層以上） |
| ステップ数 | `stop_when_decayed` 相当の処理を、`lax.scan` を固定長チャンク（例: 2000ステップ）ずつ回して判定する（§7） |

## 6. 対称性
- **z鏡映**：計算領域を z≥0 の半分にし、z=0 面に PEC（奇）または PMC（偶）境界を置く。
  - 偶（Meepの `Mirror(Z, +1)`）：PMC 境界。z=0 で `Ex, Ey, Hz` が非ゼロ。
  - 奇（`Mirror(Z, -1)`）：PEC 境界。z=0 で `Hx, Hy, Ez` が非ゼロ。
  - FDTDXに該当する境界がなければ、z=0 面で接線成分を毎ステップ0にする処理を `symmetry.py` で実装する。
  - 実装が間に合わない場合のフォールバック：z方向全体を計算し、波源をz対称（または反対称）に配置することで parity を選別する（計算量は2倍）。
- type 3〜5 の `px/py` 対称も同じ方法で実装する。type 6 は xy 方向の対称なし。

## 7. FDTD実行の仕様
- 波源：`GaussianPulseSource` 相当の点ダイポール（ガウシアン、`fcen, df, cutoff=5`）。仕様書Aと同じく、複数の点・複数の成分を同時に置き、位相は乱数にする。
- 検出器：点の時系列を記録する検出器を使う。`lax.scan` の中でGPU上のバッファに書き込み、**チャンクの終わりにまとめてホスト側へ転送する**。毎ステップの転送はしない。
- 記録の間隔：`dt_rec = dt × k`。`k` は `1/(dt_rec·f_max) ≥ 8` を満たす最大の整数（過剰サンプリングをやめる）。
- 計算終了：チャンクごとに監視点の |場|² の包絡を計算し、ピークから `1e-5` まで減衰したら停止する。上限は `T_MAX`。
- STEP 3：補正後の周波数で動く **ランニングDFT** を検出器に付け、必要な断面（z=0 の xy 面、xz 面、yz 面）の複素場を取得する。体積全体の DFT は既定で無効。
- JITコンパイル：同じ形状なら再利用する。parity と構造が変わるときだけ再コンパイルする。

## 8. Harminv CLI
```python
def run_harminv(t_series: np.ndarray, dt: float, fmin: float, fmax: float) -> pd.DataFrame:
    # 実数列なら1列、複素列なら "re im" の2列をテキストで stdin へ渡す
    cmd = ["harminv", "-t", f"{dt}", f"{fmin}-{fmax}"]
    out = subprocess.run(cmd, input=txt, capture_output=True, text=True, check=True).stdout
    # 出力の列: frequency, decay constant, Q, amplitude, phase, error
```
- 波源が切れた後（`t > t_src_end`）の時系列だけを渡す（Meep の Harminv と同じ扱い）。
- 監視点 × 成分ごとに実行する。`multiprocessing` で並列化（CPU数ぶん）。
- 後処理のフィルタ：`error < 0.1`、`Q > 20`、`|A| > Ath·max|A|`。
- 出力は現行の `read_harminv_file` と同じ列名に揃え、`my_analysis` の関数と互換にする。
- 記録点が多すぎる場合（`N > 2e5`）は、ベースバンド化と間引きをしてから渡す（予備の処理）。

## 9. 構造の移植（`structures.py`）
- `my_geometry.py` の各関数（type 3〜6）について、Meep の `GeometricObject` の並びを **SDF（符号付き距離関数）の和・差** として書き直す。
- 誘電率配列を作る：粗い計算ではセル中心で評価して二値化（`eps_averaging=False` に相当）。本計算では、各格子で 4×4×4 のサブサンプルを取り、体積分率で平均する。
- 計算は `jax.vmap` を使ってGPU上で行う（CPU側で遅くならないようにする）。
- 検証：同じ解像度で Meep の `get_array(Dielectric)` を出力し、z=0 断面の誤差を比べる（誤差 `< 2%`、境界のピクセルは除く）。

## 10. パイプラインと出力
仕様書Aの§6・§7と同じ。STEP 2 は `select_modes.py` が担当し、`best condition {date}.txt` の形式を維持する。png は matplotlib で、現行の `my_analysis` の描画を移植して出力する。

## 11. 検証計画
| 段階 | 内容 | 合格基準 |
|---|---|---|
| V1 | 誘電率配列の照合（type 3〜6） | 上記の誤差 `< 2%` |
| V2 | 単純な2Dフォトニック結晶欠陥共振器（文献値あり）で Meep と比較 | `Δf < 1e-3`、Qの相対差 `< 10%` |
| V3 | type 6 の現行条件（偶/奇）を Meep の結果と比較 | 同上 |
| V4 | Harminv CLI と Meep の内蔵 Harminv を、同じ時系列で比較 | 周波数とQがほぼ一致（`<1e-6` 相対） |
| V5 | 速度：G4 / A100 / L4 / T4 で res を変えて計測 | Mcells·step/s をログに記録。Meep版（仕様書A）より10倍以上速い |

## 12. リスク
| リスク | 対策 |
|---|---|
| FDTDX にPMC/PEC境界や点時系列検出器がない（APIの差） | `symmetry.py` で自前のマスクを実装する／検出器をカスタムで実装する |
| float32 では高Q（>1e5）の精度が足りない | 補正段階だけ float64 にする（速度は約1/2） |
| サブピクセル平均の違いによる周波数シフト | V1・V3 で補正量を測り、必要なら Meep 方式（異方性平均）を実装する |
| GPUメモリ不足（res が大きい場合） | メモリ見積もりで事前に停止する。A100 80GB を推奨 |
| Colab のセッション切れ | STEPごとに Drive へ保存し、再開できるようにする |
