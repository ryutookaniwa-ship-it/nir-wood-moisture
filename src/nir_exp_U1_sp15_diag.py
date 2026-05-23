"""
Experiment U1: P1パイプライン診断 - sp15除外LOSO分析
=====================================================
動機:
  P1 LOSO=15.4725 に対して LB=15.395 (LOSO > LB)。
  sp15が高MC(~300%)で他種と比較不能 → LOSOを引き上げている疑い。
  sp15を除外したLOSOを計算し、LBとの対応関係を確認する。

  診断結果に基づき:
  - sp15が孤立して悪いならLOSO指標の限界として受容
  - sp11など他種も高いなら改善余地あり

出力: per-species RMSE / LOSO full / LOSO ex-sp15 / OOF predictions (npy)
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

EXP = "U1"
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


# ── P1 pipeline (exact) ───────────────────────────────────────────────────────
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

print(f"=== Experiment {EXP}: P1 診断 / sp15除外LOSO ===\n")

oof_trans = np.zeros(len(y_train)); iters = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xtr[va_idx])
    iters.append(m.best_iteration)

oof = inv(oof_trans)
avg_iter = int(np.mean(iters))

# ── 診断: per-species breakdown ───────────────────────────────────────────────
print(f"{'sp':>4}  {'n':>5}  {'MC_mean':>8}  {'MC_max':>7}  {'RMSE':>8}  {'bias':>8}  {'rank':>5}")
print("-" * 56)

sp_stats = []
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    y_sp = y_train[idx]; p_sp = oof[idx]
    rmse = loso_rmse(p_sp, y_sp)
    bias = float(np.mean(p_sp - y_sp))   # mean(pred - true)
    sp_stats.append((sp, len(idx), float(y_sp.mean()), float(y_sp.max()), rmse, bias))

sp_stats.sort(key=lambda x: x[4], reverse=True)
for rank, (sp, n, mc_mean, mc_max, rmse, bias) in enumerate(sp_stats, 1):
    marker = " <<<" if sp == 15 else ""
    print(f"sp{sp:2d}  {n:5d}  {mc_mean:8.1f}  {mc_max:7.1f}  {rmse:8.4f}  {bias:+8.3f}  {rank:5d}{marker}")

# ── LOSO full vs ex-sp15 ──────────────────────────────────────────────────────
loso_full   = loso_rmse(oof, y_train)
mask_no_sp15 = sp_train != 15
loso_ex15   = loso_rmse(oof[mask_no_sp15], y_train[mask_no_sp15])

n15 = (sp_train == 15).sum()
n_total = len(y_train)

print(f"\n{'='*56}")
print(f"LOSO full          : {loso_full:.4f}  (n={n_total})")
print(f"LOSO ex-sp15       : {loso_ex15:.4f}  (n={n_total - n15})")
print(f"sp15 contribution  : {loso_full - loso_ex15:+.4f}  (sp15 n={n15}, {100*n15/n_total:.1f}%)")
print(f"avg_iter           : {avg_iter}")
print(f"LB (actual)        : 15.395  → gap = {15.395 - loso_ex15:+.4f} vs ex-sp15 LOSO")

# ── sp11 分析（2番目に悪い） ──────────────────────────────────────────────────
print(f"\n--- sp11 詳細 ---")
idx11 = np.where(sp_train == 11)[0]
print(f"  n={len(idx11)}, MC range=[{y_train[idx11].min():.1f}, {y_train[idx11].max():.1f}]")
print(f"  RMSE={loso_rmse(oof[idx11], y_train[idx11]):.4f}")
print(f"  bias={np.mean(oof[idx11] - y_train[idx11]):+.3f}")

# ── OOF保存 (U3キャリブレーション用) ─────────────────────────────────────────
import os; os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/u1_oof.npy", oof)
np.save(f"{OUT_DIR}/u1_y_train.npy", y_train)
np.save(f"{OUT_DIR}/u1_sp_train.npy", sp_train)
print(f"\nOOF saved to {OUT_DIR}/u1_oof.npy")
print(f"avg_iter for U2/U3 use: {avg_iter}")
