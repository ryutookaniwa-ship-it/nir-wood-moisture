"""
Experiment P1: ターゲット変換 細粒度グリッド探索 (p=0.22〜0.38)
==============================================================
M2: p=0.20/0.30/0.40/0.50/0.60 の5点だけ探索。
0.20と0.30の間に真の最適があるか、p=0.01刻みで細粒度探索。
ベース: M2 (p=0.30, LOSO=15.5877, LB=15.545)
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

EXP = "P1"
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

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

# M2より細かいグリッド: 0.22〜0.38
powers = [0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35, 0.38]

print(f"=== Experiment {EXP}: 細粒度 p グリッド探索 ===")
print(f"M2ベース(p=0.30, LOSO=15.5877, LB=15.545)\n")
print(f"{'p':>6}  {'LOSO':>8}  {'avg_iter':>9}  {'vs M2':>7}")
print("-" * 38)

best_rmse = np.inf; best_p = None; best_iters = None

for p in powers:
    y_trans = y_train ** p
    inv = lambda pred, p=p: np.clip(pred, 0, None) ** (1.0 / p)

    oof_trans = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    oof  = inv(oof_trans)
    rmse = loso_rmse(oof, y_train)
    avg_r = int(np.mean(iters))
    diff = rmse - 15.5877
    flag = " <-- best" if rmse < best_rmse else ""
    m2_tag = " [M2]" if abs(p - 0.30) < 0.001 else ""
    print(f"  p={p:.2f}  {rmse:8.4f}  {avg_r:9d}  {diff:+7.4f}{m2_tag}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_p = p; best_iters = iters
        best_oof_trans = oof_trans.copy()

print(f"\nBest: p={best_p:.2f}  LOSO={best_rmse:.4f}  vs M2: {best_rmse - 15.5877:+.4f}")

if best_rmse < 15.5877:
    inv_best = lambda pred, p=best_p: np.clip(pred, 0, None) ** (1.0 / p)
    y_trans_full = y_train ** best_p
    dtrain_f = lgb.Dataset(Xtr, label=y_trans_full)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv_best(final.predict(Xte))
    tag = f"p{int(best_p*100):03d}"
    OUT = f"{OUT_DIR}/submission_{EXP}_{tag}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: fine-p={best_p:.2f} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\n[Skip] M2(p=0.30)を超えなかった → 提出なし")
