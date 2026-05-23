"""
Experiment S: SG window細粒度探索 (w=5,7,9,11 + 偶数近傍)
w=9がベスト、w=7は20.27。w=5は未試行。
さらにwindow=9周辺(8は偶数でNG、7/9/11)の確認と、
polyorder=2固定でwindow=3(最小)まで探索。
Base: MSC+sqrt + R-params(lr=0.02,leaves=63,ff=0.07,mcs=10)
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

EXP_LETTER = "S"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_S_window_fine.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
y_sqrt = np.sqrt(y_train)

params = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
          'feature_fraction': 0.07, 'min_child_samples': 10}

print("=== Experiment S: SG Window Fine Search ===")
print("Known: w=9(20.10) > w=7(20.27) > w=11(20.33)")
print("Base (R): w=9  LOSO=19.68\n")

best_rmse = np.inf; best_w = 9; best_oof = None; best_Xtr = None; best_Xte = None

# polyorder=2でwindow>=3が最小。奇数のみ有効
for window in [3, 5, 7, 9, 11]:
    try:
        Xtr = sg_deriv(msc(X_train_raw, ref), window=window, polyorder=2)
        Xte = sg_deriv(msc(X_test_raw,  ref), window=window, polyorder=2)
    except Exception as e:
        print(f"  window={window:2d}  SKIP: {e}")
        continue

    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=2000,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(model.predict(Xtr[va_idx]), 0, None) ** 2

    rmse = loso_rmse(oof, y_train)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  window={window:2d}  LOSO={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_w = window
        best_oof = oof.copy(); best_Xtr = Xtr; best_Xte = Xte

print(f"\n=== RESULT ===")
print(f"LOSO-RMSE = {best_rmse:.4f}  (best window={best_w})")
print(f"Delta vs R(19.68): {best_rmse - 19.675:+.4f}")

plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp S [window={best_w}]  LOSO={best_rmse:.4f}")

dtrain = lgb.Dataset(best_Xtr, label=y_sqrt)
final  = lgb.train(params, dtrain, num_boost_round=500,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = np.clip(final.predict(best_Xte), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
