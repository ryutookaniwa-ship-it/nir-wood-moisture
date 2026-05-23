"""
Experiment S5: Test-Spectrum-Aware EPO (テストX活用EPO拡張)
============================================================
仮説:
  EPOは訓練13樹種の種間変動方向しか除去できない。
  テスト6樹種のスペクトル分布には訓練種間変動に含まれない
  追加のドメインシフト方向が存在する可能性がある。

  対策: EPO後の残差空間でX_testのPCA主成分を計算し、
        それらの方向も追加で除去する（ラベルなし使用=正当）。

アルゴリズム:
  1. 標準EPO行列 V_tr = compute_epo(Xtr, y_tr, sp)  (n_wns, 5)
  2. Xtr_epo = apply_epo(Xtr, V_tr)
     Xte_epo = apply_epo(Xte, V_tr)
  3. X_test_epo のPCA → V_te (n_wns, k)  [k=1,2,3,5を探索]
  4. 結合方向の直交化: V_combined = SVD([V_tr, V_te])[:, :n_total]
  5. Xtr_final = Xtr - Xtr @ V_combined @ V_combined.T
     Xte_final = Xte - Xte @ V_combined @ V_combined.T
  6. LGBM(P1-params) on Xtr_final

ベース: P1 (LOSO=15.4725, LB=15.395)
期待改善: -0.5〜2.0
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

EXP = "S5"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

P1_BASELINE = 15.4725
P1_LB       = 15.395

# ── Data & preprocessing ──────────────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
y_p027 = y_train ** 0.27

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}


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


def combined_removal(Xtr, Xte, V_tr, k_test):
    """
    Apply EPO from training + additional test-spectrum PCA directions.
    V_tr: (n_wns, n_tr) training EPO matrix
    k_test: number of test PCA components to additionally remove
    """
    # Apply standard EPO first
    Xtr_e = apply_epo(Xtr, V_tr)
    Xte_e = apply_epo(Xte, V_tr)

    if k_test == 0:
        return Xtr_e, Xte_e

    # Find main variation in X_test after standard EPO
    pca_te = PCA(n_components=k_test, random_state=42)
    pca_te.fit(Xte_e)
    V_te = pca_te.components_.T  # (n_wns, k_test)

    # Orthogonalize V_te against V_tr via SVD on combined matrix
    # V_all columns span the directions we want to remove
    V_all = np.hstack([V_tr, V_te])  # (n_wns, n_tr + k_test)
    # SVD of V_all: U are orthonormal basis for the column space
    U_all, s_all, _ = np.linalg.svd(V_all, full_matrices=False)
    rank = int(np.sum(s_all > 1e-10))
    V_combined = U_all[:, :rank]  # (n_wns, rank) — orthonormal basis

    # Remove combined directions
    Xtr_f = Xtr - (Xtr @ V_combined) @ V_combined.T
    Xte_f = Xte - (Xte @ V_combined) @ V_combined.T
    return Xtr_f, Xte_f


def run_loso(Xtr, y_trans, sp, params, n_rounds=3000, patience=50):
    oof = np.zeros(len(y_trans)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(patience, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)
    return oof, int(np.mean(iters))


# ── Main EPO matrix ───────────────────────────────────────────────────────────
V_tr = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)

print(f"=== Experiment {EXP}: Test-Spectrum-Aware EPO ===")
print(f"Base: P1 (LOSO={P1_BASELINE}, LB={P1_LB})")
print(f"Strategy: EPO(n=5) + Test PCA(k) additional directions\n")
print(f"{'k_test':>8}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 40)

best_rmse = np.inf; best_k = None; best_iter = None

for k_test in [0, 1, 2, 3, 5, 7, 10]:
    Xtr_f, Xte_f = combined_removal(Xtr_sg, Xte_sg, V_tr, k_test)
    oof_trans, ai = run_loso(Xtr_f, y_p027, sp_train, P1_PARAMS)
    rmse = loso_rmse(np.clip(oof_trans, 0, None) ** (1/0.27), y_train)
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  k={k_test:2d}       {rmse:8.4f}  {ai:9d}  {diff:+7.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_k = k_test; best_iter = ai
        best_Xtr = Xtr_f.copy(); best_Xte = Xte_f.copy()

print(f"\nBest: k_test={best_k}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

# ── Submission ────────────────────────────────────────────────────────────────
if best_rmse < P1_BASELINE:
    dtrain_f = lgb.Dataset(best_Xtr, label=y_p027)
    final = lgb.train(P1_PARAMS, dtrain_f,
                      num_boost_round=best_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(best_Xte), 0, None) ** (1/0.27)
    OUT = f"{OUT_DIR}/submission_{EXP}_k{best_k}.csv"
    save_submission(test_ids, preds, OUT)
    memo = f"{EXP}: TestEPO(k={best_k}) LOSO={best_rmse:.4f}"
    submit_to_signate(OUT, memo, loso=best_rmse)
else:
    print(f"\n[Skip] P1(LOSO={P1_BASELINE})を超えなかった")
    # k=0 is equivalent to P1 — submit if it confirms P1 results
    if abs(best_rmse - P1_BASELINE) < 0.05:
        print("  (k=0 result matches P1 baseline as expected)")
