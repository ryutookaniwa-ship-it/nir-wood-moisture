"""
Experiment P5: p=0.27 + sample_weight 組み合わせ
=================================================
P1: p=0.27 → LOSO=15.4725 (-0.115 vs M2)
P2: weight(MC>100%, w=2.0) → LOSO=15.5347 (-0.053 vs M2)
両者の相乗効果を検証。
探索: weight_high=[1.5, 2.0, 3.0] × thresh=[100, 150]
ベース: P1 (p=0.27, LOSO=15.4725)
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

EXP = "P5"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
POWER = 0.27

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

y_trans = y_train ** POWER
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / POWER)

print(f"=== Experiment {EXP}: p=0.27 + sample_weight 組み合わせ ===")
print(f"P1ベース(p=0.27, LOSO=15.4725)\n")
print(f"{'thresh':>8}  {'w_high':>7}  {'LOSO':>8}  {'vs P1':>7}")
print("-" * 38)

best_rmse = np.inf; best_cfg = None; best_iters = None

for thresh in [100, 150]:
    for w_high in [1.5, 2.0, 3.0]:
        weights_all = np.where(y_train > thresh, w_high, 1.0)

        oof_trans = np.zeros(len(y_train)); iters = []
        for tr_idx, va_idx, _ in loso_folds(sp_train):
            w_tr = weights_all[tr_idx]
            dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx], weight=w_tr)
            dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
            m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            oof_trans[va_idx] = m.predict(Xtr[va_idx])
            iters.append(m.best_iteration)

        oof  = inv(oof_trans)
        rmse = loso_rmse(oof, y_train)
        diff = rmse - 15.4725
        flag = " <-- best" if rmse < best_rmse else ""
        print(f"  >{thresh:3d}%  w={w_high:.1f}  {rmse:8.4f}  {diff:+7.4f}{flag}")

        if rmse < best_rmse:
            best_rmse = rmse; best_cfg = (thresh, w_high); best_iters = iters

print(f"\nBest: thresh={best_cfg[0]}%, w={best_cfg[1]}  LOSO={best_rmse:.4f}  vs P1: {best_rmse - 15.4725:+.4f}")

if best_rmse < 15.4725:
    thresh_b, w_b = best_cfg
    weights_full = np.where(y_train > thresh_b, w_b, 1.0)
    dtrain_f = lgb.Dataset(Xtr, label=y_train ** POWER, weight=weights_full)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte))
    tag = f"th{thresh_b}_w{int(w_b*10)}"
    OUT = f"{OUT_DIR}/submission_{EXP}_{tag}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: p=0.27+weight(>{thresh_b}%,w={w_b}) LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\n[Skip] P1(p=0.27)を超えなかった → 提出なし")
