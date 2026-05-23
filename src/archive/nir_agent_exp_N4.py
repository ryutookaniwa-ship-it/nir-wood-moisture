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

EXP    = "N4"
OUT    = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_N4_agent.csv"

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

# 1. Target Transformation: y^0.3
y_transformed = y_train**0.3

# 2. LOSO Cross-Validation
oof_preds_transformed = np.zeros(len(y_train))
loso_models = [] # To store models for best_iteration averaging

for tr_idx, va_idx, sp_val in loso_folds(sp_train):
    X_fold_tr, y_fold_tr = Xtr[tr_idx], y_transformed[tr_idx]
    X_fold_va, y_fold_va = Xtr[va_idx], y_transformed[va_idx]

    lgb_train_data = lgb.Dataset(X_fold_tr, y_fold_tr)
    lgb_eval_data = lgb.Dataset(X_fold_va, y_fold_va, reference=lgb_train_data)

    model = lgb.train(
        I2_PARAMS,
        lgb_train_data,
        num_boost_round=2000, # Sufficiently large number of rounds
        valid_sets=[lgb_train_data, lgb_eval_data],
        callbacks=[lgb.early_stopping(100, verbose=False)], # Early stopping
    )
    loso_models.append(model)

    # Predict on validation fold and store transformed predictions
    oof_preds_transformed[va_idx] = model.predict(X_fold_va, num_iteration=model.best_iteration)

# Inverse transform OOF predictions back to original scale
oof_preds_original_scale = oof_preds_transformed**(1/0.3)
oof_preds_original_scale[oof_preds_original_scale < 0] = 0 # Ensure non-negative

# Calculate LOSO-RMSE
rmse = loso_rmse(oof_preds_original_scale, y_train)
print(f'LOSO-RMSE = {rmse:.4f}')

# 3. Train final model on all data using the transformed target
final_lgb_train_data = lgb.Dataset(Xtr, y_transformed)

# Use average best_iteration from LOSO models for the final model
avg_best_iterations = int(np.mean([m.best_iteration for m in loso_models])) if loso_models else 1000

final_model = lgb.train(
    I2_PARAMS,
    final_lgb_train_data,
    num_boost_round=avg_best_iterations,
)

# 4. Predict on test data and inverse transform
test_preds_transformed = final_model.predict(Xte)
test_preds_original_scale = test_preds_transformed**(1/0.3)

# Ensure predictions are non-negative (moisture content cannot be negative)
test_preds_original_scale[test_preds_original_scale < 0] = 0

# 5. Save submission
save_submission(test_ids, test_preds_original_scale, OUT)

# 6. Submit to Signate
memo_text = f"ExpN4: The dominant bottleneck is sp15, characterized by catastrophic underprediction at high MC (>150%). While y^0.3 target transformation"
submit_to_signate(OUT, memo_text + f" LOSO={rmse:.4f}", loso=rmse)