"""
Experiment U3: OOF残差分析 + 線形キャリブレーション補正
=========================================================
動機:
  sp15の系統誤差を調べ、予測値のレンジに応じた補正を試みる。
  - D5 (アイソトニック回帰) は LB悪化 → 過適合
  - より単純な線形/多項式キャリブレーションを試す

アプローチ:
  A. 線形キャリブレーション: y_cal = a * pred + b (全サンプルでfit)
  B. 線形キャリブレーション: sp15除外でfit → 全サンプルに適用
  C. 二段階: sp15除外モデル + sp15専用補正係数をblend

  U1で保存したOOFを使用（または再実行）。
  最終的なテスト予測に適用して提出。

ベース: P1 (LOSO=15.4725, LB=15.395)
"""
import sys
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "U3"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_BASELINE = 15.4725


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


# ── Load OOF from U1 if available, else rerun ─────────────────────────────────
oof_path = f"{OUT_DIR}/u1_oof.npy"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

y_p027 = y_train ** 0.27
inv    = lambda pred: np.clip(pred, 0, None) ** (1.0 / 0.27)

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

if os.path.exists(oof_path):
    oof = np.load(oof_path)
    print(f"U1 OOF loaded from {oof_path}")
    # still need avg_iter for final model; run a quick re-estimate
    iters_est = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        iters_est.append(m.best_iteration)
    avg_iter = int(np.mean(iters_est))
else:
    print("U1 OOF not found → running P1 pipeline from scratch...")
    oof_trans = np.zeros(len(y_train)); iters_est = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters_est.append(m.best_iteration)
    oof = inv(oof_trans)
    avg_iter = int(np.mean(iters_est))

# ── Final base model for test predictions ────────────────────────────────────
dtrain_f = lgb.Dataset(Xtr, label=y_p027)
final_base = lgb.train(params, dtrain_f, num_boost_round=avg_iter,
                       callbacks=[lgb.log_evaluation(-1)])
test_pred_raw = inv(final_base.predict(Xte))

print(f"\n=== Experiment {EXP}: OOF残差分析 + キャリブレーション ===")
print(f"Base P1: LOSO={P1_BASELINE}\n")

# ── 残差分析 ──────────────────────────────────────────────────────────────────
residuals = oof - y_train   # pred - true
print("残差統計 (pred - true):")
print(f"  mean={residuals.mean():+.3f}  std={residuals.std():.3f}")
print(f"  sp15: mean={residuals[sp_train==15].mean():+.3f}  "
      f"std={residuals[sp_train==15].std():.3f}  "
      f"n={int((sp_train==15).sum())}")
print(f"  ex15: mean={residuals[sp_train!=15].mean():+.3f}  "
      f"std={residuals[sp_train!=15].std():.3f}")

# ── キャリブレーション手法比較 ─────────────────────────────────────────────────
print(f"\n{'手法':>30}  {'LOSO':>8}  {'LOSO_ex15':>10}  {'vs_P1':>7}")
print("-" * 60)


def eval_cal(pred_cal, label):
    loso_full = loso_rmse(pred_cal, y_train)
    loso_ex15 = loso_rmse(pred_cal[sp_train != 15], y_train[sp_train != 15])
    diff = loso_full - P1_BASELINE
    print(f"  {label:>28}  {loso_full:8.4f}  {loso_ex15:10.4f}  {diff:+7.4f}")
    return loso_full, pred_cal


results = []

# 0. P1 uncalibrated (baseline)
r, p = eval_cal(oof, "P1 uncalibrated")
results.append(("P1", r, oof.copy()))

# A. 線形キャリブレーション (全サンプル)
lr_all = LinearRegression().fit(oof.reshape(-1,1), y_train)
oof_A = np.clip(lr_all.predict(oof.reshape(-1,1)), 0, None)
te_A = np.clip(lr_all.predict(test_pred_raw.reshape(-1,1)), 0, None)
r, _ = eval_cal(oof_A, "A: Linear(all samples)")
results.append(("A", r, te_A))

# B. 線形キャリブレーション (sp15除外でfit)
mask_ex15 = sp_train != 15
lr_ex15 = LinearRegression().fit(oof[mask_ex15].reshape(-1,1), y_train[mask_ex15])
oof_B = np.clip(lr_ex15.predict(oof.reshape(-1,1)), 0, None)
te_B = np.clip(lr_ex15.predict(test_pred_raw.reshape(-1,1)), 0, None)
r, _ = eval_cal(oof_B, "B: Linear(ex-sp15 fit)")
results.append(("B", r, te_B))

# C. 2次多項式キャリブレーション (全サンプル)
from numpy.polynomial import polynomial as P
coef_C = np.polyfit(oof, y_train, 2)
oof_C = np.clip(np.polyval(coef_C, oof), 0, None)
te_C = np.clip(np.polyval(coef_C, test_pred_raw), 0, None)
r, _ = eval_cal(oof_C, "C: Poly2(all samples)")
results.append(("C", r, te_C))

# D. sp15だけ線形補正 (sp15 OOFでfit)、他はそのまま
idx15 = np.where(sp_train == 15)[0]
lr_sp15 = LinearRegression().fit(oof[idx15].reshape(-1,1), y_train[idx15])
oof_D = oof.copy()
oof_D[idx15] = np.clip(lr_sp15.predict(oof[idx15].reshape(-1,1)), 0, None)
# テスト予測: sp15補正を高MC予測値に適用 (閾値: 予測値 > sp15_min_pred)
sp15_pred_min = float(oof[idx15].min())
te_D = test_pred_raw.copy()
mask_high = test_pred_raw > sp15_pred_min
te_D[mask_high] = np.clip(lr_sp15.predict(test_pred_raw[mask_high].reshape(-1,1)), 0, None)
r, _ = eval_cal(oof_D, "D: sp15-only linear fix")
results.append(("D", r, te_D))

# ── ベスト選択 ────────────────────────────────────────────────────────────────
best = min(results[1:], key=lambda x: x[1])  # P1除く
best_label, best_rmse, best_preds = best
print(f"\nBest: {best_label}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

if best_rmse < P1_BASELINE:
    OUT = f"{OUT_DIR}/submission_{EXP}_{best_label}.csv"
    save_submission(test_ids, best_preds, OUT)
    submit_to_signate(OUT, f"{EXP}: calibration-{best_label} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("P1 baseline not beaten -> skip submission")
    print("   キャリブレーション後もLOSO改善なし。sp15の系統誤差はモデル構造で対処が必要。")
