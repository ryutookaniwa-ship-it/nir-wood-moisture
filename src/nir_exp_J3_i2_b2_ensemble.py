"""
Experiment J3: I2 × B2 アンサンブル
=================================================
I2(SG w=9,p=2+EPO n=5, LB=16.101) と
B2(SG w=5,p=3+EPO n=5, LB=17.651) を
alpha混合してLBを改善できるか？

異なるSG設定 → 予測間の相関が下がる可能性あり
alpha grid: 0.1, 0.2, ..., 0.9 (I2比率)
"""
import sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "J3"

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

params = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

ref = X_train_raw.mean(axis=0)

# === I2: SG(w=9, p=2) + EPO(n=5) ===
Xtr_i2 = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_i2 = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V_i2 = compute_epo_matrix(Xtr_i2, y_train, sp_train, n_components=5)
Xtr_i2 = apply_epo(Xtr_i2, V_i2); Xte_i2 = apply_epo(Xte_i2, V_i2)

# === B2: SG(w=5, p=3) + EPO(n=5) ===
Xtr_b2 = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
Xte_b2 = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)
V_b2 = compute_epo_matrix(Xtr_b2, y_train, sp_train, n_components=5)
Xtr_b2 = apply_epo(Xtr_b2, V_b2); Xte_b2 = apply_epo(Xte_b2, V_b2)

y_sqrt = np.sqrt(y_train)

print(f"=== Experiment {EXP}: I2×B2アンサンブル ===")
print(f"I2(LOSO=15.73) + B2(LOSO=16.44)\n")

# --- I2単体OOF ---
oof_i2 = np.zeros(len(y_train)); iters_i2 = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_i2[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(Xtr_i2[va_idx], label=y_sqrt[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_i2[va_idx] = np.clip(m.predict(Xtr_i2[va_idx]), 0, None) ** 2
    iters_i2.append(m.best_iteration)
rmse_i2 = loso_rmse(oof_i2, y_train)
print(f"I2単体 LOSO={rmse_i2:.4f}  avg_iter={int(np.mean(iters_i2))}")

# --- B2単体OOF ---
oof_b2 = np.zeros(len(y_train)); iters_b2 = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_b2[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(Xtr_b2[va_idx], label=y_sqrt[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_b2[va_idx] = np.clip(m.predict(Xtr_b2[va_idx]), 0, None) ** 2
    iters_b2.append(m.best_iteration)
rmse_b2 = loso_rmse(oof_b2, y_train)
print(f"B2単体 LOSO={rmse_b2:.4f}  avg_iter={int(np.mean(iters_b2))}")

r, _ = pearsonr(oof_i2, oof_b2)
print(f"OOF相関: r={r:.4f}\n")

# --- alpha探索 ---
print(f"{'alpha(I2)':>10}  {'LOSO':>8}")
print("-" * 25)
best_rmse = np.inf; best_alpha = 0.5
for alpha in np.arange(0.1, 1.0, 0.1):
    oof_blend = alpha * oof_i2 + (1 - alpha) * oof_b2
    rmse = loso_rmse(oof_blend, y_train)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  alpha={alpha:.1f}  {rmse:8.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_alpha = alpha

print(f"\nBest: alpha={best_alpha:.1f}  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

# --- Final prediction ---
dtrain_i2 = lgb.Dataset(Xtr_i2, label=y_sqrt)
final_i2 = lgb.train(params, dtrain_i2, num_boost_round=int(np.mean(iters_i2)),
                     callbacks=[lgb.log_evaluation(-1)])
preds_i2 = np.clip(final_i2.predict(Xte_i2), 0, None) ** 2

dtrain_b2 = lgb.Dataset(Xtr_b2, label=y_sqrt)
final_b2 = lgb.train(params, dtrain_b2, num_boost_round=int(np.mean(iters_b2)),
                     callbacks=[lgb.log_evaluation(-1)])
preds_b2 = np.clip(final_b2.predict(Xte_b2), 0, None) ** 2

preds = best_alpha * preds_i2 + (1 - best_alpha) * preds_b2
alpha_str = f"{best_alpha:.1f}".replace(".", "")
OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_J3_i2b2_a{alpha_str}.csv"
save_submission(test_ids, preds, OUT)
print(f"\n[Done] {OUT}")
