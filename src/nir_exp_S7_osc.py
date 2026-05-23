"""
Experiment S7: OSC (Orthogonal Signal Correction) + EPO + LGBM
==============================================================
仮説:
  EPOは「樹種間のスペクトル変動方向」を除去する。
  OSCは「yと無相関なX変動方向」を除去する。
  → 二者は相補的: OSC後にEPOを適用することでより純粋な含水率シグナルが残る。

OSC アルゴリズム (NIPALS法):
  1. Xとyを列方向でセンタリング
  2. 各コンポーネントについて:
     a. t = X[:, 0] で初期化
     b. 収束まで繰り返し:
        p = X.T @ t / (t @ t)  → p /= ||p||
        t_new = X @ p
        t_new = t_new - y @ (y.T @ t_new / (y.T @ y))  # yに直交化
        t = t_new
     c. デフレート: X = X - t @ p.T

Test EPO失敗の教訓:
  テストXのPCAはy信号を含む → 除去は逆効果。
  OSCはyとの相関を保ちながら非y変動のみを除去 → 安全。

前処理パイプライン (探索):
  A: OSC(k) → LGBM
  B: MSC+SG → OSC(k) → LGBM
  C: MSC+SG → OSC(k) → EPO(5) → LGBM  ← 最有望
  D: MSC+SG → EPO(5) → OSC(k) → LGBM

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

EXP = "S7"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

P1_BASELINE = 15.4725
P1_LB       = 15.395


def osc(X: np.ndarray, y: np.ndarray, n_components: int = 1,
        max_iter: int = 100, tol: float = 1e-6) -> tuple:
    """
    Orthogonal Signal Correction (NIPALS).

    Removes variation in X that is orthogonal to y.
    Returns:
        X_osc:   corrected spectra
        W, T, P: OSC weights, scores, loadings (for applying to new X)
    """
    X = X.copy().astype(np.float64)
    y_vec = y.flatten().astype(np.float64)
    # Orthogonal projector onto null space of y
    yy = y_vec @ y_vec
    P_y = np.outer(y_vec, y_vec) / yy  # projection onto y space

    W_list, T_list, P_list = [], [], []

    for _ in range(n_components):
        # Initialize score
        t = X[:, np.argmax(X.var(axis=0))]
        t = t - P_y @ t  # make orthogonal to y

        for _iter in range(max_iter):
            p = X.T @ t / (t @ t)
            p = p / np.linalg.norm(p)
            t_new = X @ p
            t_new = t_new - P_y @ t_new  # keep orthogonal to y
            if np.linalg.norm(t_new - t) < tol:
                break
            t = t_new

        w = p  # in OSC w ≈ p for NIPALS
        W_list.append(w)
        T_list.append(t)
        P_list.append(p)

        # Deflate
        X = X - np.outer(t, p)

    W = np.column_stack(W_list)  # (n_wns, n_comp)
    T = np.column_stack(T_list)  # (n_samp, n_comp)
    P = np.column_stack(P_list)  # (n_wns, n_comp)
    return X, W, T, P


def apply_osc_transform(X_new: np.ndarray, W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply fitted OSC transformation to new X."""
    X = X_new.copy().astype(np.float64)
    for i in range(W.shape[1]):
        w = W[:, i]; p = P[:, i]
        t = X @ w
        X = X - np.outer(t, p)
    return X


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


# ── Data ──────────────────────────────────────────────────────────────────────
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

V_epo = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V_epo)
Xte_epo = apply_epo(Xte_sg, V_epo)

print(f"=== Experiment {EXP}: OSC + EPO pipeline ===")
print(f"Base: P1 (LOSO={P1_BASELINE}, LB={P1_LB})\n")

best_rmse = np.inf; best_cfg = None; best_Xtr = None; best_Xte = None; best_iter = None

# ── Pipeline C: MSC+SG → OSC → EPO → LGBM  ─────────────────────────────────
print("Pipeline C: MSC+SG → OSC(k) → EPO(5) → LGBM")
print(f"{'k_osc':>7}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 42)

for k_osc in [1, 2, 3, 5, 7]:
    # OSC on MSC+SG data (using y_train for orthogonalization)
    Xtr_osc, W, T, P = osc(Xtr_sg, y_train, n_components=k_osc)
    Xte_osc = apply_osc_transform(Xte_sg, W, P)

    # EPO on OSC-processed data
    V_epo_osc = compute_epo_matrix(Xtr_osc, y_train, sp_train, n_components=5)
    Xtr_f = apply_epo(Xtr_osc, V_epo_osc)
    Xte_f = apply_epo(Xte_osc, V_epo_osc)

    oof_trans, ai = run_loso(Xtr_f, y_p027, sp_train, P1_PARAMS)
    rmse = loso_rmse(np.clip(oof_trans, 0, None) ** (1/0.27), y_train)
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  k={k_osc:2d}     {rmse:8.4f}  {ai:9d}  {diff:+7.4f}{flag}", flush=True)

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = f"C_k{k_osc}"; best_iter = ai
        best_Xtr = Xtr_f.copy(); best_Xte = Xte_f.copy()

# ── Pipeline D: MSC+SG → EPO → OSC → LGBM ─────────────────────────────────
print(f"\nPipeline D: MSC+SG → EPO(5) → OSC(k) → LGBM")
print(f"{'k_osc':>7}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 42)

for k_osc in [1, 2, 3, 5]:
    # OSC on EPO-processed data
    Xtr_f_osc, W_d, T_d, P_d = osc(Xtr_epo, y_train, n_components=k_osc)
    Xte_f_osc = apply_osc_transform(Xte_epo, W_d, P_d)

    oof_trans, ai = run_loso(Xtr_f_osc, y_p027, sp_train, P1_PARAMS)
    rmse = loso_rmse(np.clip(oof_trans, 0, None) ** (1/0.27), y_train)
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  k={k_osc:2d}     {rmse:8.4f}  {ai:9d}  {diff:+7.4f}{flag}", flush=True)

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = f"D_k{k_osc}"; best_iter = ai
        best_Xtr = Xtr_f_osc.copy(); best_Xte = Xte_f_osc.copy()

print(f"\n=== Best: {best_cfg}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f} ===")

# ── Submission ────────────────────────────────────────────────────────────────
if best_rmse < P1_BASELINE:
    dtrain_f = lgb.Dataset(best_Xtr, label=y_p027)
    final = lgb.train(P1_PARAMS, dtrain_f,
                      num_boost_round=best_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(best_Xte), 0, None) ** (1/0.27)
    OUT = f"{OUT_DIR}/submission_{EXP}_{best_cfg}.csv"
    save_submission(test_ids, preds, OUT)
    memo = f"{EXP}: OSC({best_cfg}) LOSO={best_rmse:.4f}"
    submit_to_signate(OUT, memo, loso=best_rmse)
else:
    print(f"\n[Skip] P1(LOSO={P1_BASELINE})を超えなかった")
