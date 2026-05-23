"""
Experiment P4: Tweedie 目的関数
=================================
moisture(≥0)は右裾が長い正値分布。L2より適した分布仮定でモデル化。
LightGBMのtweedie目的はlog-linkで正値を自然に扱い、
variance ∝ μ^p (p=variance_power) で裾の重さを制御。

探索パターン:
  A) Tweedie単体 (変換なし): variance_power=[1.2, 1.5, 1.8, 2.0]
  B) Tweedie + p=0.30変換:  variance_power=[1.2, 1.5, 1.8]

ベース: M2 (LOSO=15.5877, LB=15.545)
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "P4"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

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

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

base_params = {**LGBM_BASE_PARAMS,
               "objective": "tweedie", "metric": "tweedie",
               "learning_rate": 0.02, "num_leaves": 63,
               "feature_fraction": 0.07, "min_child_samples": 10}

print(f"=== Experiment {EXP}: Tweedie 目的関数 ===")
print(f"M2ベース(LOSO=15.5877, LB=15.545)\n")
print(f"{'pattern':>22}  {'LOSO':>8}  {'vs M2':>7}")
print("-" * 44)

best_rmse = np.inf; best_cfg = None; best_iters_g = None

# Pattern A: 変換なし
for vp in [1.2, 1.5, 1.8, 2.0]:
    params = {**base_params, "tweedie_variance_power": vp}
    # y_trainは正値のまま (tweedieはlog-link)
    y_safe = np.clip(y_train, 1e-3, None)

    oof = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_safe[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_safe[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    rmse = loso_rmse(oof, y_train)
    diff = rmse - 15.5877
    flag = " <-- best" if rmse < best_rmse else ""
    label = f"A: Tweedie(vp={vp})"
    print(f"  {label:>22}  {rmse:8.4f}  {diff:+7.4f}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = ("A", vp, None); best_iters_g = iters

# Pattern B: p=0.30 変換 + Tweedie
for vp in [1.2, 1.5, 1.8]:
    params = {**base_params, "tweedie_variance_power": vp}
    POWER = 0.30
    y_trans = y_train ** POWER
    y_safe_t = np.clip(y_trans, 1e-6, None)

    oof_trans = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_safe_t[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_safe_t[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    oof  = np.clip(oof_trans, 0, None) ** (1.0 / POWER)
    rmse = loso_rmse(oof, y_train)
    diff = rmse - 15.5877
    flag = " <-- best" if rmse < best_rmse else ""
    label = f"B: p=0.30+Tw(vp={vp})"
    print(f"  {label:>22}  {rmse:8.4f}  {diff:+7.4f}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = ("B", vp, POWER); best_iters_g = iters

print(f"\nBest: pattern={best_cfg[0]}, vp={best_cfg[1]}  LOSO={best_rmse:.4f}  vs M2: {best_rmse - 15.5877:+.4f}")

if best_rmse < 15.5877:
    pat, vp_b, p_b = best_cfg
    final_params = {**base_params, "tweedie_variance_power": vp_b}
    if p_b:
        y_label = np.clip(y_train ** p_b, 1e-6, None)
        inv_f = lambda pred: np.clip(pred, 0, None) ** (1.0 / p_b)
    else:
        y_label = np.clip(y_train, 1e-3, None)
        inv_f = lambda pred: pred

    dtrain_f = lgb.Dataset(Xtr, label=y_label)
    final = lgb.train(final_params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters_g)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv_f(final.predict(Xte))
    tag = f"pat{pat}_vp{int(vp_b*10)}"
    OUT = f"{OUT_DIR}/submission_{EXP}_{tag}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: Tweedie(vp={vp_b}) pat={pat} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\n[Skip] M2を超えなかった → 提出なし")
