"""
Experiment RL2: DART (Dropouts meet Multiple Additive Regression Trees)
========================================================================
P3: rounds=800, rate_drop=0.05, p=0.30 → LOSO=23.07 (+7.49) 大幅悪化
原因仮説: rounds不足 (DARTは通常のGBDTより多くのroundsが必要)

RL2: rounds=2000, rate_drop=[0.05, 0.10, 0.20], p=0.27 で再探索
DARTはearly_stopping非対応のため固定rounds。
P1のavg_iter≈600を参考に2000を上限とする。
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP = "RL2"
P_POWER = 0.27
ROUNDS = 2000


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T


def apply_epo(X, V): return X - (X @ V) @ V.T


train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
y_train  = train[target_col].values
X_tr_raw = train[spec_cols].values.astype(np.float64)
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_train = train["species number"].values
y_pow    = y_train ** P_POWER

ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

RATE_DROPS = [0.05, 0.10, 0.20]

print(f"=== {EXP}: DART rate_drop探索 (rounds={ROUNDS}, p=0.27) ===")
print(f"{'rate_drop':>10}  {'LOSO':>8}  {'delta':>7}")
print("-" * 35)

best_rmse = np.inf; best_rd = None

for rd in RATE_DROPS:
    params = {
        **LGBM_BASE_PARAMS,
        "boosting_type": "dart",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "feature_fraction": 0.07,
        "min_child_samples": 10,
        "drop_rate": rd,
        "skip_drop": 0.5,
        "max_drop": 50,
        "uniform_drop": False,
    }
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_pow[tr_idx])
        m = lgb.train(params, dtrain, num_boost_round=ROUNDS,
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** (1 / P_POWER)

    rmse = loso_rmse(oof, y_train)
    tag = " <-- best" if rmse < best_rmse else ""
    print(f"  rd={rd:.2f}    {rmse:8.4f}  {rmse-15.4725:+7.4f}{tag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_rd = rd

print(f"\nBest DART: rate_drop={best_rd}, LOSO={best_rmse:.4f}")
print(f"P1 (GBDT): LOSO=15.4725")
print(f"Delta: {best_rmse-15.4725:+.4f}")

best_params = {
    **LGBM_BASE_PARAMS,
    "boosting_type": "dart",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
    "drop_rate": best_rd,
    "skip_drop": 0.5,
    "max_drop": 50,
}
OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
dtrain_f = lgb.Dataset(Xtr_epo, label=y_pow)
final = lgb.train(best_params, dtrain_f, num_boost_round=ROUNDS,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: DART rd={best_rd}, LOSO={best_rmse:.4f}", loso=best_rmse)
print(f"[Done] {EXP}")
