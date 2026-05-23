"""
Experiment P: SG窓サイズ最適化
前提: window=11は仮定値。7/9/13/15/21を探索。
Base: MSC+sqrt(y)+H-params (Exp M LOSO=20.33)
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_lgbm,
    save_submission, plot_residuals, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "P"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_P_sg_window.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
h_params = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
            'feature_fraction': 0.07, 'min_child_samples': 30}
y_sqrt = np.sqrt(y_train)

print("=== Experiment P: SG Window Size ===")
print("Base (M): window=11  LOSO=20.33\n")

best_rmse = np.inf; best_w = 11; best_oof = None; best_Xtr = None; best_Xte = None

for window in [7, 9, 11, 13, 15, 21]:
    Xtr = sg_deriv(msc(X_train_raw, ref), window=window)
    Xte = sg_deriv(msc(X_test_raw,  ref), window=window)

    from nir_loso_utils import loso_folds, loso_rmse
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        model  = lgb.train(h_params, dtrain, num_boost_round=2000,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(model.predict(Xtr[va_idx]), 0, None) ** 2

    rmse = loso_rmse(oof, y_train)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  window={window:2d}  LOSO-RMSE={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_w = window
        best_oof = oof.copy(); best_Xtr = Xtr; best_Xte = Xte

print(f"\n=== RESULT ===")
print(f"LOSO-RMSE = {best_rmse:.4f}  (best window={best_w})")
print(f"Delta vs M(20.33): {best_rmse - 20.3336:+.4f}")

plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp P [window={best_w}]  LOSO={best_rmse:.4f}")

dtrain = lgb.Dataset(best_Xtr, label=np.sqrt(y_train))
final  = lgb.train(h_params, dtrain, num_boost_round=400,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = np.clip(final.predict(best_Xte), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
