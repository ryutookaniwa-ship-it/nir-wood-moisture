"""
Experiment U2: EPO拡張 - sp15 within-species PCA方向を追加
===========================================================
動機:
  標準EPO (bin_width=10, min_species=2) はMC>100%のビンをスキップする。
  理由: そのビンに含まれる樹種がsp15のみ(1種) → inter-species方向なし。
  結果: EPOはsp15の高MC領域の樹種固有パターンを全く除去していない。

解決策:
  sp15のwithin-species PCA (top k成分) をEPOの除去方向に追加する。
  sp15の「自種内スペクトル変動」の主要方向を除去することで、
  sp15の高MC領域での樹種固有パターンを軽減する。

  V_combined = SVD([V_standard | V_sp15_pca])[:n_final]

リスク:
  sp15の水分情報が含まれる方向も除去してしまう可能性あり。
  → k_sp15 を 2/3/5 で探索。

ベース: P1 (LOSO=15.4725, LB=15.395)
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

EXP = "U2"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_BASELINE = 15.4725


def compute_epo_matrix_standard(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
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
    return Vt[:n_components].T   # (1555, 5)


def compute_epo_extended(X, y, sp, k_sp15=3, n_final=5):
    """Standard EPO + sp15 within-species PCA."""
    V_std = compute_epo_matrix_standard(X, y, sp, n_components=n_final)

    # sp15 within-species PCA
    idx15 = np.where(sp == 15)[0]
    X15 = X[idx15] - X[idx15].mean(axis=0)   # center within sp15
    k_sp15_eff = min(k_sp15, len(idx15) - 1)
    pca15 = PCA(n_components=k_sp15_eff, random_state=42)
    pca15.fit(X15)
    V_sp15 = pca15.components_   # (k_sp15, 1555)

    # Combine and re-orthogonalize via SVD
    D_all = np.vstack([V_std.T, V_sp15])   # (5+k_sp15, 1555)
    _, _, Vt = np.linalg.svd(D_all, full_matrices=False)
    return Vt[:n_final].T   # (1555, n_final)


def apply_epo(X, V): return X - (X @ V) @ V.T


# ── Data ──────────────────────────────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)

y_p027 = y_train ** 0.27
inv    = lambda pred: np.clip(pred, 0, None) ** (1.0 / 0.27)

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

print(f"=== Experiment {EXP}: EPO + sp15 PCA拡張 ===")
print(f"Base: P1(LOSO={P1_BASELINE})\n")
print(f"{'k_sp15':>7}  {'LOSO_full':>9}  {'LOSO_ex15':>10}  {'avg_iter':>9}  {'vs_P1':>7}")
print("-" * 52)

best_rmse = np.inf; best_k = None; best_iters_list = None

for k_sp15 in [2, 3, 5]:
    V = compute_epo_extended(Xtr_sg, y_train, sp_train, k_sp15=k_sp15, n_final=5)
    Xtr = apply_epo(Xtr_sg, V)
    Xte = apply_epo(Xte_sg, V)

    oof_trans = np.zeros(len(y_train)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    oof = inv(oof_trans)
    rmse_full = loso_rmse(oof, y_train)
    mask_ex15 = sp_train != 15
    rmse_ex15 = loso_rmse(oof[mask_ex15], y_train[mask_ex15])
    avg_r = int(np.mean(iters))
    diff = rmse_full - P1_BASELINE
    flag = " <-- best" if rmse_full < best_rmse else ""
    print(f"  k={k_sp15:2d}    {rmse_full:9.4f}  {rmse_ex15:10.4f}  {avg_r:9d}  {diff:+7.4f}{flag}")

    if rmse_full < best_rmse:
        best_rmse = rmse_full; best_k = k_sp15; best_iters_list = iters[:]
        best_V = V.copy(); best_oof = oof.copy()

print(f"\nBest: k_sp15={best_k}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

# per-species for best k
print("\nPer-species RMSE (best k):")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    print(f"  sp{sp:2d}: RMSE={loso_rmse(best_oof[idx], y_train[idx]):.4f}")

if best_rmse < P1_BASELINE:
    Xtr_best = apply_epo(Xtr_sg, best_V)
    Xte_best = apply_epo(Xte_sg, best_V)
    dtrain_f = lgb.Dataset(Xtr_best, label=y_p027)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters_list)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte_best))
    OUT = f"{OUT_DIR}/submission_{EXP}_k{best_k}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: EPO+sp15PCA(k={best_k}) LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\nP1 baseline not beaten -> skip submission")
    print("   sp15 PCA方向の除去が含水率シグナルも除去している可能性")
