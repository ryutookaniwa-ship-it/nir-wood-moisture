"""
Experiment M: log(y+1) target transform -> LGBM -> exp(pred)-1
仮説: 含水率は0-298%と広範囲。高MC(>100%)を系統的に過小予測している(EDAで確認済み)。
log変換でターゲットを圧縮すると、高MC領域の予測精度が改善する可能性がある。
MSC+SG1+H-params (Exp L LOSO=20.54) をベースに検証。
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")

from nir_loso_utils import (
    load_data, snv, sg_deriv, msc,
    loso_folds, loso_rmse,
    save_submission, plot_residuals,
    LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP_LETTER = "M"
OUT_PATH   = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_M_log_target.csv"

data = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

# ── 前処理 (L best: MSC+SG1) ─────────────────────────────────────────────────
ref  = X_train_raw.mean(axis=0)
X_tr = sg_deriv(msc(X_train_raw, ref))
X_te = sg_deriv(msc(X_test_raw,  ref))

# ── H-params ─────────────────────────────────────────────────────────────────
h_params = {**LGBM_BASE_PARAMS,
            'learning_rate': 0.02,
            'num_leaves': 63,
            'feature_fraction': 0.07,
            'min_child_samples': 30}

print("=== Experiment M: log(y+1) target transform ===")
print("Base (L): MSC+SG1+H-params  LOSO=20.54\n")

def loso_with_log(X_tr, y, sp_train, params, use_log=True, n_rounds=2000):
    """LOSO-CV with optional log transform of target."""
    y_fit = np.log1p(y) if use_log else y
    oof_transformed = np.zeros(len(y))

    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_fit[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y_fit[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof_transformed[va_idx] = model.predict(X_tr[va_idx])

    # 元スケールに戻す
    oof = np.expm1(oof_transformed) if use_log else oof_transformed
    oof = np.clip(oof, 0, None)
    rmse = loso_rmse(oof, y)
    return rmse, oof

# ── 比較実験 ─────────────────────────────────────────────────────────────────
print("--- log変換あり vs なし (MSC+SG1+H-params) ---")
rmse_no_log, oof_no_log = loso_with_log(X_tr, y_train, sp_train, h_params, use_log=False)
print(f"  No log transform:  LOSO-RMSE={rmse_no_log:.4f}")

rmse_log, oof_log = loso_with_log(X_tr, y_train, sp_train, h_params, use_log=True)
print(f"  log(y+1) transform: LOSO-RMSE={rmse_log:.4f}")

# ── sqrt変換も試す ────────────────────────────────────────────────────────────
print("\n--- sqrt変換 ---")
def loso_with_sqrt(X_tr, y, sp_train, params, n_rounds=2000):
    y_fit = np.sqrt(y)
    oof_sq = np.zeros(len(y))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_fit[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y_fit[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof_sq[va_idx] = model.predict(X_tr[va_idx])
    oof = np.clip(oof_sq ** 2, 0, None)
    return loso_rmse(oof, y), oof

rmse_sqrt, oof_sqrt = loso_with_sqrt(X_tr, y_train, sp_train, h_params)
print(f"  sqrt(y) transform: LOSO-RMSE={rmse_sqrt:.4f}")

# ── 結果まとめ ────────────────────────────────────────────────────────────────
results = {
    "no_log": (rmse_no_log, oof_no_log),
    "log1p":  (rmse_log,    oof_log),
    "sqrt":   (rmse_sqrt,   oof_sqrt),
}
best_label = min(results, key=lambda k: results[k][0])
best_rmse, best_oof = results[best_label]

print(f"\n=== RESULT ===")
print(f"LOSO-RMSE = {best_rmse:.4f}  (best: {best_label})")
print(f"Baseline  : 21.48")
print(f"Best (L)  : 20.54")
print(f"Delta vs G: {best_rmse - 21.48:+.4f}")
print(f"Delta vs L: {best_rmse - 20.54:+.4f}")

# ── プロット ─────────────────────────────────────────────────────────────────
plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp M [{best_label}]  LOSO={best_rmse:.4f}")

# ── 最終モデル ────────────────────────────────────────────────────────────────
use_log_final  = (best_label == "log1p")
use_sqrt_final = (best_label == "sqrt")

y_fit = (np.log1p(y_train) if use_log_final
         else np.sqrt(y_train) if use_sqrt_final
         else y_train)

dtrain = lgb.Dataset(X_tr, label=y_fit)
final  = lgb.train(h_params, dtrain, num_boost_round=400,
                   callbacks=[lgb.log_evaluation(-1)])
preds_raw = final.predict(X_te)
preds = (np.expm1(preds_raw) if use_log_final
         else preds_raw ** 2  if use_sqrt_final
         else preds_raw)

save_submission(test_ids, preds, OUT_PATH)
