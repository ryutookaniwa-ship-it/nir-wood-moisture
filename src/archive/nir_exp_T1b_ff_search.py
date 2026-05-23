"""
Experiment T1b: T1 (w=[5,9,13]) + ff グリッド探索
===================================================
T1でsp3が3000rounds上限到達 → ff=0.023が低すぎてunderfitting疑い。
ff=0.04/0.05/0.06/0.07でT1パイプラインを再実行。
ベース: P1 (LOSO=15.4725), T1 (LOSO=16.0680, ff=0.023)
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

EXP = "T1b"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_BASELINE = 15.4725
WINDOWS = [5, 9, 13]


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


# ── Data & pipeline (same as T1) ─────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
Xtr_msc = msc(X_train_raw, ref)
Xte_msc = msc(X_test_raw, ref)

Xtr_concat = np.hstack([sg_deriv(Xtr_msc, window=w, polyorder=2) for w in WINDOWS])
Xte_concat = np.hstack([sg_deriv(Xte_msc, window=w, polyorder=2) for w in WINDOWS])
print(f"concat shape: {Xtr_concat.shape}")

V = compute_epo_matrix(Xtr_concat, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_concat, V)
Xte = apply_epo(Xte_concat, V)

y_p027 = y_train ** 0.27
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / 0.27)

n_feat = Xtr.shape[1]  # 4665

print(f"=== Experiment {EXP}: T1 (w={WINDOWS}) ff grid search ===")
print(f"Base: P1(LOSO={P1_BASELINE}), T1(LOSO=16.0680, ff=0.023)\n")
print(f"{'ff':>6}  {'feat/tree':>9}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 48)

best_rmse = np.inf; best_ff = None; best_iters = None

for ff in [0.04, 0.05, 0.06, 0.07]:
    params = {**LGBM_BASE_PARAMS,
              "learning_rate": 0.02, "num_leaves": 63,
              "feature_fraction": ff, "min_child_samples": 10}

    oof_trans = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    oof = inv(oof_trans); rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(iters))
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  ff={ff:.2f}  {int(n_feat*ff):9d}  {rmse:8.4f}  {avg_r:9d}  {diff:+7.4f}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_ff = ff; best_iters = iters[:]

print(f"\nBest: ff={best_ff}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

if best_rmse < P1_BASELINE:
    params_best = {**LGBM_BASE_PARAMS,
                   "learning_rate": 0.02, "num_leaves": 63,
                   "feature_fraction": best_ff, "min_child_samples": 10}
    dtrain_f = lgb.Dataset(Xtr, label=y_p027)
    final = lgb.train(params_best, dtrain_f,
                      num_boost_round=int(np.mean(best_iters)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte))
    OUT = f"{OUT_DIR}/submission_{EXP}_ff{int(best_ff*100):03d}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: T1 ff={best_ff} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("P1 baseline not beaten -> skip submission")
