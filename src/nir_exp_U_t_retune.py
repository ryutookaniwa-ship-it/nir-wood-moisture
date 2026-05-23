"""
Experiment U: T設定(w=5,poly=3)でハイパラ再チューニング
T: MSC+SG(w=5,poly=3)+sqrt  R-params(lr=0.02,leaves=63,ff=0.07,mcs=10)  LOSO=19.547
window/polyorderが変わるとfeature_fraction等の最適値も変化する可能性。
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

EXP_LETTER = "U"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_U_t_retune.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
X_tr   = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
X_te   = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)
y_sqrt = np.sqrt(y_train)

base = dict(objective='regression', metric='rmse', verbosity=-1,
            n_jobs=-1, random_state=42, learning_rate=0.02)

print("=== Experiment U: Retune on T-preprocessing (w=5, poly=3) ===")
print("T baseline: LOSO=19.547\n")

best_rmse = np.inf; best_cfg = None; best_oof = None

def cv(params, n_rounds=2000):
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(model.predict(X_tr[va_idx]), 0, None) ** 2
    return loso_rmse(oof, y_train), oof

# Phase 1: ff x leaves
print("--- Phase 1: feature_fraction x num_leaves ---")
for ff in [0.05, 0.07, 0.10, 0.15]:
    for leaves in [31, 63, 127]:
        params = {**base, 'num_leaves': leaves, 'feature_fraction': ff, 'min_child_samples': 10}
        rmse, oof = cv(params)
        flag = " <-- best" if rmse < best_rmse else ""
        print(f"  ff={ff:.2f} leaves={leaves:3d}  LOSO={rmse:.4f}{flag}")
        if rmse < best_rmse:
            best_rmse = rmse; best_cfg = params.copy(); best_oof = oof.copy()

# Phase 2: min_child_samples
print(f"\n--- Phase 2: min_child_samples ---")
for mcs in [5, 10, 20, 30]:
    params = {**best_cfg, 'min_child_samples': mcs}
    rmse, oof = cv(params)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  mcs={mcs:2d}  LOSO={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = params.copy(); best_oof = oof.copy()

# Phase 3: learning_rate
print(f"\n--- Phase 3: learning_rate ---")
for lr, n in [(0.02, 2000), (0.01, 3000), (0.005, 5000)]:
    params = {**best_cfg, 'learning_rate': lr}
    rmse, oof = cv(params, n_rounds=n)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  lr={lr:.3f}  LOSO={rmse:.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = params.copy(); best_oof = oof.copy()

print(f"\n=== RESULT ===")
print(f"Best config: {best_cfg}")
print(f"LOSO-RMSE = {best_rmse:.4f}")
print(f"Delta vs T(19.55): {best_rmse - 19.547:+.4f}")
print(f"Delta vs R(19.68): {best_rmse - 19.675:+.4f}")

plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp U [w=5,poly=3 retune]  LOSO={best_rmse:.4f}")

dtrain = lgb.Dataset(X_tr, label=y_sqrt)
final  = lgb.train(best_cfg, dtrain, num_boost_round=500,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = np.clip(final.predict(X_te), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
