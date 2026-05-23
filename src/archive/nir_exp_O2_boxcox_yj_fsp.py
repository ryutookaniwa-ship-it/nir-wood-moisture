"""Experiment O2-O4: Box-Cox / Yeo-Johnson / FSP-split transform"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")

from nir_loso_utils import (
    load_data, sg_deriv, msc,
    loso_folds, loso_rmse,
    save_submission, plot_residuals,
    LGBM_BASE_PARAMS,
)
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP_LETTER = "O"
OUT_PATH   = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_O_transform.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref  = X_train_raw.mean(axis=0)
X_tr = sg_deriv(msc(X_train_raw, ref))
X_te = sg_deriv(msc(X_test_raw,  ref))

h_params = {**LGBM_BASE_PARAMS, 'learning_rate': 0.02, 'num_leaves': 63,
            'feature_fraction': 0.07, 'min_child_samples': 30}

print("=== O2-O4: Box-Cox / Yeo-Johnson / FSP-split ===")
print("O1 result (power grid best): p=0.50 (sqrt)  LOSO=20.3336\n")

results = {"sqrt(p=0.50)": (20.3336, None, {"type": "power", "p": 0.50})}

# ── O2: Box-Cox ───────────────────────────────────────────────────────────────
print("--- O2: Box-Cox ---")
oof_bc = np.zeros(len(y_train))
lambdas = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    y_t, lam = stats.boxcox(y_train[tr_idx])
    lambdas.append(lam)
    y_va_t = stats.boxcox(y_train[va_idx], lmbda=lam)
    dtrain = lgb.Dataset(X_tr[tr_idx], label=y_t)
    dval   = lgb.Dataset(X_tr[va_idx], label=y_va_t, reference=dtrain)
    model  = lgb.train(h_params, dtrain, num_boost_round=2000,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
    oof_bc[va_idx] = np.clip(inv_boxcox(model.predict(X_tr[va_idx]), lam), 0, None)

rmse_bc = loso_rmse(oof_bc, y_train)
mean_lam = np.mean(lambdas)
print(f"  mean_lambda={mean_lam:.3f}  LOSO-RMSE={rmse_bc:.4f}")
results["BoxCox"] = (rmse_bc, oof_bc, {"type": "boxcox", "lambda": mean_lam})

# ── O3: Yeo-Johnson ───────────────────────────────────────────────────────────
print("\n--- O3: Yeo-Johnson ---")
oof_yj = np.zeros(len(y_train))
yj_lambdas = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    y_t   = pt.fit_transform(y_train[tr_idx].reshape(-1,1)).ravel()
    y_va_t = pt.transform(y_train[va_idx].reshape(-1,1)).ravel()
    yj_lambdas.append(pt.lambdas_[0])
    dtrain = lgb.Dataset(X_tr[tr_idx], label=y_t)
    dval   = lgb.Dataset(X_tr[va_idx], label=y_va_t, reference=dtrain)
    model  = lgb.train(h_params, dtrain, num_boost_round=2000,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
    pred_t = model.predict(X_tr[va_idx])
    oof_yj[va_idx] = np.clip(pt.inverse_transform(pred_t.reshape(-1,1)).ravel(), 0, None)

rmse_yj = loso_rmse(oof_yj, y_train)
print(f"  mean_lambda={np.mean(yj_lambdas):.3f}  LOSO-RMSE={rmse_yj:.4f}")
results["YeoJohnson"] = (rmse_yj, oof_yj, {"type": "yeojohnson", "lambda": np.mean(yj_lambdas)})

# ── O4: FSP分割変換 ───────────────────────────────────────────────────────────
print("\n--- O4: FSP-aware piecewise (MC<30:sqrt, MC>=30:log) ---")
FSP = 30.0

def fsp_fwd(y):
    return np.where(y < FSP, np.sqrt(y),
                    np.log1p(y - FSP) + np.sqrt(FSP)), None

def fsp_inv(pred, _):
    thr = np.sqrt(FSP)
    return np.clip(np.where(pred < thr, pred**2,
                             np.expm1(pred - thr) + FSP), 0, None)

oof_fsp = np.zeros(len(y_train))
for tr_idx, va_idx, sp in loso_folds(sp_train):
    y_t, _ = fsp_fwd(y_train[tr_idx])
    y_va_t, _ = fsp_fwd(y_train[va_idx])
    dtrain = lgb.Dataset(X_tr[tr_idx], label=y_t)
    dval   = lgb.Dataset(X_tr[va_idx], label=y_va_t, reference=dtrain)
    model  = lgb.train(h_params, dtrain, num_boost_round=2000,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
    oof_fsp[va_idx] = fsp_inv(model.predict(X_tr[va_idx]), None)

rmse_fsp = loso_rmse(oof_fsp, y_train)
print(f"  LOSO-RMSE={rmse_fsp:.4f}")
results["FSP-split"] = (rmse_fsp, oof_fsp, {"type": "fsp"})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== FULL SUMMARY (O1+O2+O3+O4) ===")
print(f"{'Transform':<20} {'LOSO-RMSE':>10}  {'vs sqrt':>8}")
for label, (rmse, _, cfg) in sorted(results.items(), key=lambda x: x[1][0]):
    delta = rmse - 20.3336
    marker = " <-- BEST" if rmse == min(r[0] for r in results.values()) else ""
    print(f"  {label:<18} {rmse:>10.4f}  {delta:>+8.4f}{marker}")

best_label, (best_rmse, best_oof, best_cfg) = min(results.items(), key=lambda x: x[1][0])
print(f"\nLOSO-RMSE = {best_rmse:.4f}  (best: {best_label})")
print(f"Delta vs G(21.48): {best_rmse - 21.48:+.4f}")
print(f"Delta vs M(20.33): {best_rmse - 20.3336:+.4f}")

if best_oof is not None:
    plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
                   title=f"Exp O [{best_label}]  LOSO={best_rmse:.4f}")

# ── 最終モデル ────────────────────────────────────────────────────────────────
cfg = best_cfg
if cfg["type"] == "power":
    p = cfg["p"]
    y_fit = y_train ** p
    inv_final = lambda pred: np.clip(pred, 0, None) ** (1.0 / p)
elif cfg["type"] == "boxcox":
    lam = cfg["lambda"]
    y_fit = stats.boxcox(y_train, lmbda=lam)
    inv_final = lambda pred: np.clip(inv_boxcox(pred, lam), 0, None)
elif cfg["type"] == "yeojohnson":
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    y_fit = pt.fit_transform(y_train.reshape(-1,1)).ravel()
    inv_final = lambda pred: np.clip(pt.inverse_transform(pred.reshape(-1,1)).ravel(), 0, None)
elif cfg["type"] == "fsp":
    y_fit, _ = fsp_fwd(y_train)
    inv_final = lambda pred: fsp_inv(pred, None)

dtrain = lgb.Dataset(X_tr, label=y_fit)
final  = lgb.train(h_params, dtrain, num_boost_round=400,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = inv_final(final.predict(X_te))
save_submission(test_ids, preds, OUT_PATH)
