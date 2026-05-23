"""
Experiment I: Water absorption band focused spectrum -> LGBM
樹種固有パターンを排除し、普遍的な水の吸収信号に集中する

水の主要吸収帯:
  ~5187 cm-1: O-H伸縮+変角 結合音 (最重要)
  ~6896 cm-1: O-H伸縮 第1倍音 (最重要)
  ~8333 cm-1: O-H伸縮 第2倍音 (中)
  6707-7082 cm-1: 水素結合状態変動域 (高)

戦略:
  I1. 水バンド周辺のみ切り出し -> LGBM
  I2. 水バンド幅を変えて探索
  I3. 全スペクトルから木材成分バンドを除外 -> LGBM

LGBM baseline (G): LOSO=21.48, LB=18.995
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/train (1).csv"
TEST_PATH  = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/test (2).csv"
OUT_PATH   = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/output/submission_I_waterbands.csv"

# ── Load ───────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding='shift-jis')
test  = pd.read_csv(TEST_PATH,  encoding='shift-jis')

target_col  = train.columns[3]
spec_cols   = train.columns[4:].tolist()
wns         = np.array([float(c) for c in spec_cols])

y_train     = train[target_col].values
X_train_raw = train[spec_cols].values.astype(np.float64)
X_test_raw  = test[spec_cols].values.astype(np.float64)
test_ids    = test['sample number'].values
sp_train    = train['species number'].values

# ── Preprocessing ──────────────────────────────────────────────────────────────
def snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / np.where(s == 0, 1, s)

def sg_deriv(X, window=11, polyorder=2, deriv=1):
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=deriv, axis=1)

X_tr_full = sg_deriv(snv(X_train_raw))
X_te_full = sg_deriv(snv(X_test_raw))

# ── LOSO-CV helper ─────────────────────────────────────────────────────────────
def loso_folds(sp):
    for s in sorted(set(sp)):
        yield np.where(sp != s)[0], np.where(sp == s)[0]

lgbm_params = dict(
    objective='regression', metric='rmse', verbosity=-1,
    n_jobs=-1, random_state=42,
    learning_rate=0.05, num_leaves=31,
    feature_fraction=0.1, min_child_samples=10,
)

def loso_lgbm(X_tr, X_te_dummy, y, params, n_rounds=500):
    oof = np.zeros(len(y))
    for tr_idx, va_idx in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict(X_tr[va_idx])
    return np.sqrt(np.mean((y - oof) ** 2)), oof

# ── 帯域選択ユーティリティ ──────────────────────────────────────────────────────
def select_bands(X_tr, X_te, wns, bands):
    """bands: list of (lo, hi) cm-1"""
    mask = np.zeros(len(wns), dtype=bool)
    for lo, hi in bands:
        mask |= (wns >= lo) & (wns <= hi)
    print(f"    Selected {mask.sum()} / {len(wns)} wavenumbers")
    return X_tr[:, mask], X_te[:, mask]

print("=== Experiment I: Water band focused spectrum -> LGBM ===")
print(f"Baseline LOSO-RMSE: 21.48  LB: 18.995\n")

best_rmse  = np.inf
best_label = None
best_X_tr  = None
best_X_te  = None

# ── I1: 主要水バンドのみ（幅を変えて探索）────────────────────────────────────
print("--- I1: Primary water bands only ---")

# 水バンドの組み合わせを試す
band_configs = {
    "5187±200 + 6896±300": [(4987,5387),(6596,7196)],
    "5187±200 + 6896±300 + 8333±200": [(4987,5387),(6596,7196),(8133,8533)],
    "5187±300 + 6707-7082 + 8333±200": [(4887,5487),(6707,7082),(8133,8533)],
    "水バンド全域 4800-7300": [(4800,7300)],
    "水バンド全域 4800-9000": [(4800,9000)],
    "5187±100 + 6896±150 (narrow)": [(5087,5287),(6746,7046)],
    "5000-7500 cm-1": [(5000,7500)],
    "4500-8500 cm-1": [(4500,8500)],
}

for label, bands in band_configs.items():
    Xtr_b, Xte_b = select_bands(X_tr_full, X_te_full, wns, bands)
    rmse, _ = loso_lgbm(Xtr_b, Xte_b, y_train, lgbm_params)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  [{label}]  LOSO-RMSE={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse  = rmse
        best_label = label
        best_X_tr  = Xtr_b
        best_X_te  = Xte_b

# ── I2: 木材成分バンドを除外したスペクトル ────────────────────────────────────
print("\n--- I2: Full spectrum EXCLUDING wood component bands ---")
# 木材成分主要干渉帯を除外
wood_bands = [
    (4700, 4820),   # セルロース O-H+C-H (~4760)
    (4350, 4450),   # C-H 結合音 (~4400)
    (5850, 6050),   # C-H 第1倍音 (~5950)
    (5550, 5850),   # リグニン芳香族C-H (~5700)
]
exclude_mask = np.zeros(len(wns), dtype=bool)
for lo, hi in wood_bands:
    exclude_mask |= (wns >= lo) & (wns <= hi)
include_mask = ~exclude_mask
print(f"    Excluded {exclude_mask.sum()} wavenumbers (wood bands), kept {include_mask.sum()}")

Xtr_ex = X_tr_full[:, include_mask]
Xte_ex = X_te_full[:, include_mask]
rmse, _ = loso_lgbm(Xtr_ex, Xte_ex, y_train, lgbm_params)
flag = " <-- best" if rmse < best_rmse else ""
print(f"  [Full - wood bands]  LOSO-RMSE={rmse:.4f}{flag}")
if rmse < best_rmse:
    best_rmse  = rmse
    best_label = "Full - wood bands"
    best_X_tr  = Xtr_ex
    best_X_te  = Xte_ex

# ── I3: 全スペクトル (feature_fraction を大きくしても水バンドが選ばれるか) ───
print("\n--- I3: Full spectrum with higher feature_fraction ---")
for ff in [0.05, 0.2, 0.3]:
    p = {**lgbm_params, 'feature_fraction': ff}
    rmse, _ = loso_lgbm(X_tr_full, X_te_full, y_train, p)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  [full, feat_frac={ff}]  LOSO-RMSE={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse  = rmse
        best_label = f"full feat_frac={ff}"
        best_X_tr  = X_tr_full
        best_X_te  = X_te_full
        lgbm_params_best = {**p}

# ── Summary & Final predictions ────────────────────────────────────────────────
print(f"\n=== RESULT ===")
print(f"Best: [{best_label}]  LOSO-RMSE={best_rmse:.4f}")
print(f"Baseline LGBM      : 21.48")
print(f"Delta              : {best_rmse - 21.48:+.4f}")

# 最良設定で全データ学習→テスト予測
use_params = lgbm_params if 'lgbm_params_best' not in dir() else lgbm_params_best
dtrain = lgb.Dataset(best_X_tr, label=y_train)
final_model = lgb.train(use_params, dtrain, num_boost_round=500,
                        callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final_model.predict(best_X_te), 0, None)

import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
pd.DataFrame({'id': test_ids, 'pred': preds}).to_csv(OUT_PATH, index=False, header=False)
print(f"Saved: {OUT_PATH}")
