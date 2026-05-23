"""
Experiment T: SG polyorder探索
polyorder=2を前提としてきたが、1(線形)/3/4を試す。
window=9固定(R best)でpolyorderを変える。
さらに (window, polyorder) の組み合わせも探索。
Base (R): MSC+SG(w=9,poly=2)+sqrt  LOSO=19.68
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

EXP_LETTER = "T"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_T_polyorder.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
y_sqrt = np.sqrt(y_train)

params = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
          'feature_fraction': 0.07, 'min_child_samples': 10}

print("=== Experiment T: SG Polyorder Search ===")
print("Base (R): window=9, polyorder=2  LOSO=19.68\n")

best_rmse = np.inf; best_cfg = None; best_oof = None; best_Xtr = None; best_Xte = None

# Phase 1: window=9固定でpolyorderを変える
print("--- Phase 1: window=9, polyorder varied ---")
for polyorder in [1, 2, 3, 4]:
    # polyorder < window が必要
    if polyorder >= 9:
        continue
    try:
        Xtr = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=polyorder)
        Xte = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=polyorder)
    except Exception as e:
        print(f"  poly={polyorder}  SKIP: {e}")
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
    print(f"  window=9  poly={polyorder}  LOSO={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = (9, polyorder)
        best_oof = oof.copy(); best_Xtr = Xtr; best_Xte = Xte

# Phase 2: 有望なpolyorderで他のwindowも試す
best_poly = best_cfg[1]
if best_poly != 2:
    print(f"\n--- Phase 2: poly={best_poly} x window grid ---")
    for window in [5, 7, 9, 11, 13]:
        if best_poly >= window:
            continue
        try:
            Xtr = sg_deriv(msc(X_train_raw, ref), window=window, polyorder=best_poly)
            Xte = sg_deriv(msc(X_test_raw,  ref), window=window, polyorder=best_poly)
        except Exception as e:
            print(f"  window={window}  SKIP: {e}")
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
        print(f"  window={window:2d}  poly={best_poly}  LOSO={rmse:.4f}{flag}")
        if rmse < best_rmse:
            best_rmse = rmse; best_cfg = (window, best_poly)
            best_oof = oof.copy(); best_Xtr = Xtr; best_Xte = Xte
else:
    print(f"\nPhase 2 skip: polyorder=2 already best (same as R)")

print(f"\n=== RESULT ===")
print(f"Best: window={best_cfg[0]}, polyorder={best_cfg[1]}")
print(f"LOSO-RMSE = {best_rmse:.4f}")
print(f"Delta vs R(19.68): {best_rmse - 19.675:+.4f}")

plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp T [w={best_cfg[0]},poly={best_cfg[1]}]  LOSO={best_rmse:.4f}")

dtrain = lgb.Dataset(best_Xtr, label=y_sqrt)
final  = lgb.train(params, dtrain, num_boost_round=500,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = np.clip(final.predict(best_Xte), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
