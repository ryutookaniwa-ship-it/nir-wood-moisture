"""
Experiment K3: XGBoost + I2パイプライン
=================================================
I2(MSC+SG(w=9,p=2)+EPO(n=5)+LGBM, LB=16.101)のLGBMを
XGBoostに置き換えて汎化性能を比較。

XGBoostとLGBMの主な違い:
  - level-wise成長 (LGBMはleaf-wise)
  - L1/L2正則化が明示的
  - colsample_bytree = LGBMのfeature_fraction相当

1点打ち (グリッドサーチなし):
  LGBMの最適値(ff=0.07, num_leaves=63相当→max_depth=6)をそのまま移植
  → LOSO1周分(13fold)のみで完結
前処理固定: MSC+SG(w=9,p=2) → EPO(n=5) → sqrt(y)
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission,
)
import xgboost as xgb
import warnings; warnings.filterwarnings("ignore")

EXP = "K3"

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

print(f"=== Experiment {EXP}: XGBoost + I2パイプライン (1点打ち) ===")
print(f"I2(LGBM, LOSO=15.73, LB=16.101)との比較")
print(f"params: max_depth=6, colsample_bytree=0.07 (LGBMの最適値を移植)\n")

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.02,
    "max_depth": 6,
    "colsample_bytree": 0.07,
    "subsample": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

oof = np.zeros(len(y_train)); best_iters = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = xgb.DMatrix(Xtr_epo[tr_idx], label=y_sqrt[tr_idx])
    dval   = xgb.DMatrix(Xtr_epo[va_idx], label=y_sqrt[va_idx])
    m = xgb.train(
        params, dtrain, num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof[va_idx] = np.clip(m.predict(dval), 0, None) ** 2
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train); avg_bi = int(np.mean(best_iters))
print(f"LOSO-RMSE : {rmse:.4f}")
print(f"vs I2(LGBM): {rmse - 15.73:+.4f}")
print(f"avg_iter  : {avg_bi}")

dtrain_f = xgb.DMatrix(Xtr_epo, label=y_sqrt)
final = xgb.train(params, dtrain_f, num_boost_round=avg_bi, verbose_eval=False)
preds = np.clip(final.predict(xgb.DMatrix(Xte_epo)), 0, None) ** 2
OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_K3_xgb_d6_c007.csv"
save_submission(test_ids, preds, OUT)
print(f"\n[Done] {OUT}")
