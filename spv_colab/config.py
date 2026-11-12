import json
import math
import os
import numpy as np

# ==============================================================================
# §4. ユーザー入力パラメータ
# ==============================================================================
cfg = dict(
    structure_type = 6,
    structure = dict(            # 構造ごとに必要なキーだけを書く
        # type 6: n, a, r_ratio, slab_ratio, N_ring, twist_deg, core_r_ratio, gap_ratio, bowtie_angle
        # type 3-5: n, period, Disk_R, w, Disk_t, Sac_t, Sac_D, Out_D, Gap, Wg_w, Wg_t, Cir_reso, Dev_rate
        n            = 3.4,
        a            = 0.45,
        r_ratio      = 0.26,
        slab_ratio   = 0.50,
        N_ring       = 8,
        twist_deg    = 3.5,
        core_r_ratio = 0.55,
        gap_ratio    = 0.02,
        bowtie_angle = 60.0,
    ),
    f_range   = (0.55, 0.75),     # 探索帯域（Meep単位 a/λ）
    parities  = ['+', '-'],       # 計算するz対称（偶/奇）
    accuracy  = dict(coarse=2, fine=4),  # 「最小特徴寸法あたりの格子数」（粗探索/本計算）
    n_modes   = 2,                # 本計算で扱う最大モード数
    run_mode  = 'ALL',            # ALL / HARMINV / MODE（既存ログの再利用）
)

# オプション制御フラグ (§6, §9)
MERGE_REFINE_MODE = True          # STEP 2.5 で同時に DFT 場を取得し、Qが十分高ければ STEP 3 を省略
N_SHOTS = 1                       # 同時励起の乱数シード実行回数（既定値 1）
RANDOM_SEED = 42                  # 同時励起の乱数位相シード（固定）
SAVE_H5_FILES = False             # 大容量 .h5 書き出しの省略
OUT_DIR = os.environ.get('SPV_OUT_DIR', './')

# 内部定数 (§5)
RES_MAX = 120
CUTOFF = 5.0
DECAY_DT = 50
DECAY_BY = 1e-5
T_MAX_COARSE = 2000
ATH = 1e-5
Q_MIN = 20.0
REFINE_MATCH_FACTOR = 0.6
Q_MERGE_THRESHOLD = 100.0         # MERGE_REFINE_MODE 時に STEP 3 を省略する最小Q値

# 構造タイプ別の既定パラメータ（未指定キーの補完用）
DEFAULT_STRUCTURES = {
    6: dict(
        n=3.4, a=0.45, r_ratio=0.26, slab_ratio=0.50, N_ring=8,
        twist_deg=3.5, core_r_ratio=0.55, gap_ratio=0.02, bowtie_angle=60.0,
    ),
    3: dict(
        n=2.4, period=1.5, Disk_R=1.0, w=0.3, Disk_t=0.14,
        Sac_t=0.5, Sac_D=0.0, Out_D=10.0, Gap=0.12, Wg_w=0.155, Wg_t=0.14,
        Cir_reso=200, Dev_rate=0.0,
    ),
    4: dict(
        n=2.4, period=1.5, Disk_R=1.0, w=0.3, Disk_t=0.14,
        Sac_t=0.5, Sac_D=0.0, Out_D=10.0, Gap=0.12, Wg_w=0.155, Wg_t=0.14,
        Cir_reso=200, Dev_rate=0.0,
    ),
    5: dict(
        n=2.4, period=1.5, Disk_R=1.0, w=0.3, Disk_t=0.14,
        Sac_t=0.5, Sac_D=0.0, Out_D=10.0, Gap=0.12, Wg_w=0.155, Wg_t=0.14,
        Cir_reso=200, Dev_rate=0.0,
    ),
}

PARITY_MAP = {'+': 1, '-': -1, 'o': 0, 1: '+', -1: '-', 0: 'o'}


# ==============================================================================
# §5. 自動で導出するパラメータ関数群
# ==============================================================================
def get_merged_structure(user_cfg):
    """構造種別に応じた既定値とユーザー指定の structure 辞書をマージする"""
    stype = int(user_cfg['structure_type'])
    if stype not in DEFAULT_STRUCTURES:
        raise ValueError(f"未対応の structure_type です: {stype} (対応: 3, 4, 5, 6)")
    merged = dict(DEFAULT_STRUCTURES[stype])
    merged.update(user_cfg.get('structure', {}))
    if stype == 6:
        if 'period' in merged and 'a' not in user_cfg.get('structure', {}):
            merged['a'] = merged['period']
        merged['period'] = merged['a']
    else:
        if 'a' in merged and 'period' not in user_cfg.get('structure', {}):
            merged['period'] = merged['a']
        merged['a'] = merged['period']
    return merged


def min_feature(user_cfg):
    """最小特徴寸法 d_min を計算する (§5)"""
    stype = int(user_cfg['structure_type'])
    st = get_merged_structure(user_cfg)
    if stype == 6:
        return float(st['gap_ratio']) * float(st['a'])
    elif stype == 5:
        return float(min(st['Gap'], st['Wg_w'], st['Wg_t'], st['Disk_t']))
    elif stype in (3, 4):
        return float(min(st['w'], st['Disk_t']))
    else:
        raise ValueError(f"未対応の structure_type です: {stype}")


def calc_cell_size(user_cfg, dist_to_pml):
    """構造寸法 + dist_to_pml から計算セル sp = [xm, ym, zm] を計算する (§5)"""
    stype = int(user_cfg['structure_type'])
    st = get_merged_structure(user_cfg)
    period = float(st['period'])
    if stype == 6:
        slab_t = round(float(st['slab_ratio']) * period, 3)
        xy_span = 2.0 * float(st['N_ring']) * period + 2.0 * dist_to_pml[0]
        zm = slab_t + 2.0 * dist_to_pml[2]
        return [xy_span, xy_span, zm]
    elif stype in (3, 4, 5):
        disk_t = float(st['Disk_t'])
        return [period + dist_to_pml[0], period + dist_to_pml[1], disk_t + dist_to_pml[2]]
    else:
        raise ValueError(f"未対応の structure_type です: {stype}")


def get_monitor_fields(pz):
    """z対称性に応じた監視・励起3成分を返す (§5)"""
    pz_val = PARITY_MAP[pz] if isinstance(pz, str) else int(pz)
    if pz_val == 1:
        # 偶対称 (pz=+1): z=0面で非ゼロになる成分
        return ['ex', 'ey', 'hz']
    elif pz_val == -1:
        # 奇対称 (pz=-1): z=0面で非ゼロになる成分
        return ['hx', 'hy', 'ez']
    else:
        return ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']


def get_default_monitor_positions(user_cfg):
    """構造ごとの既定監視点リスト m_pos_harminv (period比座標) を返す (§5)"""
    stype = int(user_cfg['structure_type'])
    st = get_merged_structure(user_cfg)
    if stype == 6:
        slab_ratio = float(st['slab_ratio'])
        return [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.15, 0.10, 0.0],
            [0.0, 0.0, 0.25 * slab_ratio],
        ]
    else:
        disk_t = float(st['Disk_t'])
        period = float(st['period'])
        ratio = disk_t / period
        return [
            [0.0, 0.475, 0.0 * ratio],
            [0.0, 0.475, 0.45 * ratio],
            [ratio / 2.0, 0.475, 0.0 * ratio],
            [ratio / 2.0, 0.475, 0.45 * ratio],
            [ratio, 0.475, 0.0 * ratio],
            [ratio, 0.475, 0.45 * ratio],
            [0.475, 0.0, 0.0 * ratio],
            [0.475, 0.0, 0.45 * ratio],
            [0.475, ratio / 2.0, 0.0 * ratio],
            [0.475, ratio / 2.0, 0.45 * ratio],
            [0.475, ratio, 0.0 * ratio],
            [0.475, ratio, 0.45 * ratio],
        ]


def get_symmetry_combinations(user_cfg, pz_char):
    """構造ごとの (px, py, pz) 対称性タプルのリストを返す (§5)"""
    stype = int(user_cfg['structure_type'])
    pz_val = PARITY_MAP[pz_char] if isinstance(pz_char, str) else int(pz_char)
    if stype == 6:
        # type 6: xy方向は対称なし (0, 0)、zは parities で指定
        return [[0, 0, pz_val]]
    else:
        # type 3〜5: 現行の px/py 指定ロジック (parity=[-1, 1, 0], px_indices=0..1, py_indices=1..1)
        return [[-1, 1, pz_val], [1, 1, pz_val]]


def calc_df_refine(f, Q_coarse):
    """STEP 2.5 狭帯域再測定の df_refine = clip(10*f/Q_coarse, 0.01, 0.06) (§5)"""
    q_safe = max(float(Q_coarse), 1.0)
    return float(np.clip(10.0 * float(f) / q_safe, 0.01, 0.06))


def calc_df_mode(f, Q_refine):
    """STEP 3 モード計算の df_mode = clip(2*f/Q_refine, 0.005, 0.025) (§5)"""
    q_safe = max(float(Q_refine), 1.0)
    return float(np.clip(2.0 * float(f) / q_safe, 0.005, 0.025))


def calc_tmax_fine(f, Q_est):
    """補正・本計算の上限時間 T_MAX = 5*Q_est/f (§5)"""
    f_safe = max(float(f), 1e-4)
    q_safe = max(float(Q_est), 20.0)
    return float(max(50.0, round(5.0 * q_safe / f_safe, 1)))


def derive_params(user_cfg=None, verbose=True):
    """
    ユーザー設定辞書 cfg から §5 の全自動導出パラメータを計算して辞書として返す。
    """
    if user_cfg is None:
        user_cfg = cfg

    stype = int(user_cfg['structure_type'])
    st = get_merged_structure(user_cfg)
    n = float(st['n'])
    period = float(st['period'])

    f_min, f_max = float(user_cfg['f_range'][0]), float(user_cfg['f_range'][1])
    if f_min <= 0 or f_max <= f_min:
        raise ValueError(f"f_range が不正です: {user_cfg['f_range']}")

    # 1. 最小特徴寸法 d_min と 解像度 res_harminv, res_mode
    d_min = min_feature(user_cfg)
    d_min_over_period = d_min / period
    res_floor = math.ceil(12.0 * n * f_max)

    acc_coarse = float(user_cfg['accuracy']['coarse'])
    acc_fine = float(user_cfg['accuracy']['fine'])

    res_harminv = max(math.ceil(acc_coarse / d_min_over_period), res_floor)
    res_mode = max(math.ceil(acc_fine / d_min_over_period), res_floor)

    if verbose:
        if res_harminv > RES_MAX:
            print(f"[警告] 粗探索解像度 res_harminv={res_harminv} が上限 RES_MAX={RES_MAX} を超えているため {RES_MAX} に制限します。")
        if res_mode > RES_MAX:
            print(f"[警告] 本計算解像度 res_mode={res_mode} が上限 RES_MAX={RES_MAX} を超えているため {RES_MAX} に制限します。")
    res_harminv = min(res_harminv, RES_MAX)
    res_mode = min(res_mode, RES_MAX)

    # 2. 波源中心周波数 fcen, 帯域幅 df, cutoff
    fcen = 0.5 * (f_min + f_max)
    df = 1.2 * (f_max - f_min)
    cutoff = CUTOFF

    # 3. dist_to_pml と pml_w
    half_wavelength = 0.5 / f_min
    legacy_z_dist = 0.55 if stype == 6 else 0.60
    legacy_pml_w = 0.35 if stype == 6 else 0.10

    dist_xy = round(half_wavelength, 4)
    dist_z = round(max(half_wavelength, legacy_z_dist), 4)
    dist_to_pml = [dist_xy, dist_xy, dist_z]

    pml_val = round(max(half_wavelength, legacy_pml_w), 4)
    pml_w = [pml_val, pml_val, pml_val]

    # 4. セルサイズ sp
    sp = calc_cell_size(user_cfg, dist_to_pml)

    # 5. 波源位置（ギャップ/構造中心 [0,0,0] と 0.05a ずらした点の2か所同時励起）
    # ※ type 3〜5 では中心に穴がある構造や対称面を考慮し、必要に応じて user_cfg['src_pos'] で上書き可能
    default_src_pos = [
        [0.0, 0.0, 0.0],
        [0.05, 0.05, 0.0],
    ]
    if stype in (3, 4, 5) and 'src_pos' not in user_cfg:
        # ディスク・リングでは外周付近にもモードが局在するため中心/0.05a に加えて外周代表点でも励起可能だが、
        # §5「ギャップ（構造の中心）と、そこから 0.05a ずらした点の2か所に同時に置く」に準拠しつつ
        # ディスクモードが確実に励起されるよう半径方向オフセットもサポート
        default_src_pos = [
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.0],
        ]
    src_pos_harminv = np.array(user_cfg.get('src_pos', default_src_pos), dtype=float)

    # 6. 監視点
    m_pos_harminv = user_cfg.get('m_pos', get_default_monitor_positions(user_cfg))

    derived = dict(
        structure_type = stype,
        structure      = st,
        n              = n,
        period         = period,
        d_min          = d_min,
        f_min          = f_min,
        f_max          = f_max,
        fcen           = fcen,
        df             = df,
        cutoff         = cutoff,
        res_harminv    = int(res_harminv),
        res_mode       = int(res_mode),
        dist_to_pml    = dist_to_pml,
        pml_w          = pml_w,
        sp             = sp,
        parities       = list(user_cfg.get('parities', ['+', '-'])),
        n_modes        = int(user_cfg.get('n_modes', 2)),
        run_mode       = str(user_cfg.get('run_mode', 'ALL')).upper(),
        src_pos        = src_pos_harminv,
        m_pos          = m_pos_harminv,
        tmax_coarse    = int(user_cfg.get('tmax_coarse', T_MAX_COARSE)),
        decay_dt       = DECAY_DT,
        decay_by       = DECAY_BY,
        Ath            = ATH,
        Q_min          = Q_MIN,
        refine_tol_fac = REFINE_MATCH_FACTOR,
        merge_refine   = bool(user_cfg.get('merge_refine_mode', MERGE_REFINE_MODE)),
        n_shots        = int(user_cfg.get('n_shots', N_SHOTS)),
        seed           = int(user_cfg.get('seed', RANDOM_SEED)),
        save_h5        = bool(user_cfg.get('save_h5', SAVE_H5_FILES)),
        out_dir        = str(user_cfg.get('out_dir', OUT_DIR)),
    )
    return derived


_RUNTIME_CFG_PATH = os.environ.get(
    'SPV_USER_CFG_JSON',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.runtime_cfg.json')
)


def save_runtime_cfg(user_cfg=None, path=None):
    """ノートブック上で変更した cfg を MPI サブプロセス (run_sim.py) へ引き継ぐために保存する"""
    target_cfg = cfg if user_cfg is None else user_cfg
    target_path = _RUNTIME_CFG_PATH if path is None else path
    serializable = dict(target_cfg)
    if 'f_range' in serializable:
        serializable['f_range'] = list(serializable['f_range'])
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return target_path


if os.path.exists(_RUNTIME_CFG_PATH):
    try:
        with open(_RUNTIME_CFG_PATH, 'r', encoding='utf-8') as _f:
            _loaded = json.load(_f)
        if 'f_range' in _loaded:
            _loaded['f_range'] = tuple(_loaded['f_range'])
        cfg.update(_loaded)
    except Exception:
        pass

