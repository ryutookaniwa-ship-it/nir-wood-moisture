"""Experiment N2: ```json
{
  "hypothesis": "The dominant bottleneck is sp15, characterized by catastrophic underprediction at high MC values (>150%). While y^0.30 target transformation"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src/nir")

from nir_loso_utils import (
    load_data, msc, snv, sg_deriv,
    loso_folds, loso_rmse, loso_lgbm, loso_sklearn,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP    = "N2"
OUT    = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_N2_agent.csv"

# ── I2ベースパイプライン (MSC+SG(w=9,p=2)+EPO(n=5)) ────────────────────────
def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width; mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species: continue
        sp_means = np.array([X[mask][sp[mask]==s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0]-1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T

def apply_epo(X, V): return X - (X @ V) @ V.T

# I2固定パラメータ
I2_PARAMS = {**LGBM_BASE_PARAMS,
              "learning_rate": 0.02, "num_leaves": 63,
              "feature_fraction": 0.07, "min_child_samples": 10}

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref   = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

# ===== EXPERIMENT CODE BELOW =====
# 注意事項:
#   - ターゲット変換は y_sqrt = np.sqrt(y_train) が I2 デフォルト
#   - LOSO評価は loso_folds(sp_train) を使う
#   - 提出前評価: print(f"LOSO-RMSE = {rmse:.4f}")  ← この形式必須
#   - save_submission + submit_to_signate(OUT, memo, loso=rmse) で提出
#     (LOSO >= 15.4 なら自動的にスキップされる)
#   - I2_PARAMS と Xtr, Xte, y_train, sp_train はすでに定義済み
#   - 新しいモデルや前処理の試みはこのセクションに書く
# ===== END INSTRUCTIONS =====

# Target transformation as per hypothesis
TARGET_POWER = 0.30
y_transformed = y_train**TARGET_POWER

oof_preds = np.zeros_like(y_train, dtype=float)
best_iterations = []

# Perform Leave-One-Species-Out Cross-Validation
for tr_idx, va_idx, sp_val in loso_folds(sp_train):
    X_tr_fold, X_va_fold = Xtr[tr_idx], Xtr[va_idx]
    y_tr_fold, y_va_fold = y_transformed[tr_idx], y_transformed[va_idx]

    lgb_train = lgb.Dataset(X_tr_fold, y_tr_fold)
    lgb_eval = lgb.Dataset(X_va_fold, y_va_fold, reference=lgb_train)

    model = lgb.train(
        I2_PARAMS,
        lgb_train,
        num_boost_round=2000, # Increased max rounds for robustness
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(100, verbose=False)] # Early stopping
    )
    best_iterations.append(model.best_iteration)

    # Predict on validation set and inverse transform
    preds_va_transformed = model.predict(X_va_fold)
    oof_preds[va_idx] = preds_va_transformed**(1/TARGET_POWER)

# Calculate LOSO-RMSE using the original y_train
rmse = loso_rmse(oof_preds, y_train)
print(f'LOSO-RMSE = {rmse:.4f}')

# Train final model on the entire dataset with the transformed target
avg_best_rounds = int(np.mean(best_iterations)) if best_iterations else 1000 # Fallback if no early stopping occurred
final_lgb_train = lgb.Dataset(Xtr, y_transformed)
final_model = lgb.train(
    I2_PARAMS,
    final_lgb_train,
    num_boost_round=avg_best_rounds
)

# Predict on test data and inverse transform
preds_test_transformed = final_model.predict(Xte)
preds_test = preds_test_transformed**(1/TARGET_POWER)

# Save submission file
save_submission(test_ids, preds_test, OUT)

# Submit to Signate
submit_to_signate(OUT, f'Exp{EXP}: y^{TARGET_POWER:.2f} target transform. LOSO={rmse:.4f}', loso=rmse)