"""
Experiment RL3: bagging_fraction (行サンプリング) 探索
======================================================
P1は列サンプリング(ff=0.07)のみ使用。行サンプリング未使用。
bagging_fraction + bagging_freq=1 を追加することで
ツリーごとの多様性が増し汎化が改善するか検証。

グリッド: bagging_fraction=[0.7, 0.8, 0.9] vs 1.0(P1相当)
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

EXP = "RL3"
P_POWER = 0.27

BASE_PARAMS = {**LGBM_BASE_PARAMS,
               "learning_rate": 0.02, "num_leaves": 63,
               "feature_fraction": 0.07, "min_child_samples": 10}


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

FRACS = [1.0, 0.9, 0.8, 0.7]

print(f"=== {EXP}: bagging_fraction探索 ===")
print(f"{'bag_frac':>9}  {'LOSO':>8}  {'avg_iter':>9}  {'delta':>7}")
print("-" * 42)

best_rmse = np.inf; best_frac = None; best_iters_list = None

for frac in FRACS:
    params = {**BASE_PARAMS}
    if frac < 1.0:
        params["bagging_fraction"] = frac
        params["bagging_freq"] = 1
    oof = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_pow[tr_idx])
        dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_pow[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** (1 / P_POWER)
        iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(iters))
    tag = " [P1]" if frac == 1.0 else (" <-- best" if rmse < best_rmse else "")
    print(f"  {frac:9.1f}  {rmse:8.4f}  {avg_r:9d}  {rmse-15.4725:+7.4f}{tag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_frac = frac; best_iters_list = iters

print(f"\nBest: bagging_fraction={best_frac}, LOSO={best_rmse:.4f}")

best_params = {**BASE_PARAMS}
if best_frac < 1.0:
    best_params["bagging_fraction"] = best_frac
    best_params["bagging_freq"] = 1
OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
avg_r = int(np.mean(best_iters_list))
dtrain_f = lgb.Dataset(Xtr_epo, label=y_pow)
final = lgb.train(best_params, dtrain_f, num_boost_round=avg_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: bag={best_frac}, LOSO={best_rmse:.4f}", loso=best_rmse)
print(f"[Done] {EXP}")
