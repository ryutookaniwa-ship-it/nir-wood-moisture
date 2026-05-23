"""
Experiment V: Blend R + M (and variants)
R: MSC+SG(w=9)+sqrt  LGBM(lr=0.02,leaves=63,ff=0.07,mcs=10)  LOSO=19.68 LB=18.403
M: MSC+SG(w=11)+sqrt LGBM(lr=0.02,leaves=63,ff=0.07,mcs=30)  LOSO=20.33 LB=18.723

OOFをLOSO-CV内で再生成し、元スケールでブレンド最適化。
ブレンドは元スケール（%）で実施 → sqrt空間でのブレンドより自然。
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, plot_residuals, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "V"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_V_blend.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
y_sqrt = np.sqrt(y_train)

# ── 各設定の前処理 ────────────────────────────────────────────────────────────
X_tr_R = sg_deriv(msc(X_train_raw, ref), window=9)   # R: w=9
X_te_R = sg_deriv(msc(X_test_raw,  ref), window=9)

X_tr_M = sg_deriv(msc(X_train_raw, ref), window=11)  # M: w=11
X_te_M = sg_deriv(msc(X_test_raw,  ref), window=11)

params_R = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
            'feature_fraction': 0.07, 'min_child_samples': 10}

params_M = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
            'feature_fraction': 0.07, 'min_child_samples': 30}

print("=== Experiment V: Blend R + M ===")
print("R: w=9, mcs=10  LOSO=19.68  LB=18.403")
print("M: w=11, mcs=30 LOSO=20.33  LB=18.723\n")

# ── OOF生成 ───────────────────────────────────────────────────────────────────
def get_oof(X_tr, y_sqrt, sp_train, params, n_rounds=2000):
    oof = np.zeros(len(y_sqrt))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(model.predict(X_tr[va_idx]), 0, None)
    return oof  # sqrt空間

print("Generating OOF for R...")
oof_R_sqrt = get_oof(X_tr_R, y_sqrt, sp_train, params_R)
oof_R = oof_R_sqrt ** 2
rmse_R = loso_rmse(oof_R, y_train)
print(f"  R OOF LOSO-RMSE: {rmse_R:.4f}")

print("Generating OOF for M...")
oof_M_sqrt = get_oof(X_tr_M, y_sqrt, sp_train, params_M)
oof_M = oof_M_sqrt ** 2
rmse_M = loso_rmse(oof_M, y_train)
print(f"  M OOF LOSO-RMSE: {rmse_M:.4f}")

# ── OOF相関確認 ──────────────────────────────────────────────────────────────
residuals_R = oof_R - y_train
residuals_M = oof_M - y_train
corr = np.corrcoef(residuals_R, residuals_M)[0, 1]
print(f"\nResidual correlation R vs M: {corr:.4f}")
print(f"(低いほどブレンド効果が大きい)")

# ── ブレンド比探索 (元スケールで) ────────────────────────────────────────────
print("\n--- Blend ratio search (alpha*R + (1-alpha)*M) ---")
best_rmse = np.inf; best_alpha = 1.0

for alpha in np.arange(0.0, 1.05, 0.05):
    blend = alpha * oof_R + (1 - alpha) * oof_M
    rmse  = loso_rmse(np.clip(blend, 0, None), y_train)
    flag  = " <-- best" if rmse < best_rmse else ""
    print(f"  alpha={alpha:.2f}  LOSO={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_alpha = alpha

print(f"\n=== RESULT ===")
print(f"Best alpha (R weight): {best_alpha:.2f}")
print(f"LOSO-RMSE = {best_rmse:.4f}")
print(f"Delta vs R(19.68): {best_rmse - 19.675:+.4f}")
print(f"Delta vs G(21.48): {best_rmse - 21.48:+.4f}")

best_oof = np.clip(best_alpha * oof_R + (1 - best_alpha) * oof_M, 0, None)
plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp V [alpha={best_alpha:.2f}*R + {1-best_alpha:.2f}*M]  LOSO={best_rmse:.4f}")

# ── テスト予測生成 ────────────────────────────────────────────────────────────
print("\nGenerating test predictions...")
def get_test_pred(X_tr, X_te, y_sqrt, params, n_rounds=500):
    dtrain = lgb.Dataset(X_tr, label=y_sqrt)
    model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                       callbacks=[lgb.log_evaluation(-1)])
    return np.clip(model.predict(X_te), 0, None) ** 2

pred_R = get_test_pred(X_tr_R, X_te_R, y_sqrt, params_R)
pred_M = get_test_pred(X_tr_M, X_te_M, y_sqrt, params_M)
preds  = np.clip(best_alpha * pred_R + (1 - best_alpha) * pred_M, 0, None)
save_submission(test_ids, preds, OUT_PATH)
