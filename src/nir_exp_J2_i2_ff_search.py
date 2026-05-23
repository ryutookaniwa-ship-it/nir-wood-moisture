"""
Experiment J2: I2ベース feature_fraction再探索
=================================================
I2(SG w=9,p=2+EPO n=5, LB=16.101)をベースに
feature_fractionの最適値を再探索。
I1ではB2ベース(SG w=5,p=3)でff=0.07が最適。
SG設定が変わった今、最適ffも変わるか？

グリッドサーチ:
  feature_fraction: 0.02, 0.03, 0.05, 0.07(I2), 0.10, 0.15
前処理: MSC+SG(w=9,p=2) → EPO(n=5) → sqrt(y) → LGBM
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "J2"

def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width; mask = (y >= lo) & (y < hi)
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

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
Xtr_pp = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_pp = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V = compute_epo_matrix(Xtr_pp, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_pp, V); Xte_epo = apply_epo(Xte_pp, V)
y_sqrt = np.sqrt(y_train)

base_params = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "num_leaves": 63,
               "min_child_samples": 10}

print(f"=== Experiment {EXP}: I2ベース feature_fraction探索 ===")
print(f"I2(ff=0.07, LOSO=15.73, LB=16.101)との比較\n")
print(f"{'ff':>6}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 30)

best_rmse = np.inf; best_ff = None; best_bi = None

for ff in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
    params = {**base_params, "feature_fraction": ff}
    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    i2_tag = " [I2]" if ff == 0.07 else ""
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  ff={ff:.2f}  {rmse:8.4f}  {avg_r:9d}{i2_tag}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_ff = ff; best_bi = best_iters

print(f"\nBest: ff={best_ff}  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

params_f = {**base_params, "feature_fraction": best_ff}
dtrain_f = lgb.Dataset(Xtr_epo, label=y_sqrt)
final = lgb.train(params_f, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** 2
ff_str = f"{best_ff:.2f}".replace(".", "")
OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_J2_i2_ff{ff_str}.csv"
save_submission(test_ids, preds, OUT)
print(f"\n[Done] {OUT}")
