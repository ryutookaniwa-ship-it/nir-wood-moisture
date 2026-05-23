"""
Experiment T1: Multi-scale SG + Joint EPO
==========================================
仮説:
  SG(w=9)は特定の周波数帯の微分情報しか捉えない。
  w=5（高周波）・w=9（中周波）・w=13（低周波）を並列に適用し、
  3スケールの微分スペクトルをconcatすることで表現力を向上させる。

設計（Option B: Joint EPO）:
  各w → MSC → SG(w, poly=2) → concat (4665-dim)
  → EPO_joint(n=5) fit on 4665-dim joint space → LGBM

feature_fraction の再設計:
  P1: 1555 × 0.07 = 108.85 特徴量/tree
  T1: 4665 × 0.023 = 107.3  特徴量/tree  ← 同等の多様性を維持

ベース: P1 (LOSO=15.4725, LB=15.395)
期待: 3スケール情報の追加で+0.3〜0.5改善
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

EXP = "T1"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_BASELINE = 15.4725

WINDOWS = [5, 9, 13]
FF = 0.023   # 4665 * 0.023 ≈ 107.3 features/tree (P1: 1555 * 0.07 = 108.9)


# ── EPO helpers ───────────────────────────────────────────────────────────────
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


# ── Data & multi-scale preprocessing ─────────────────────────────────────────
data = load_data()
y_train    = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids   = data["test_ids"]
sp_train   = data["sp_train"]

# MSC reference: computed from training data only (no leakage)
ref = X_train_raw.mean(axis=0)
Xtr_msc = msc(X_train_raw, ref)
Xte_msc = msc(X_test_raw,  ref)

# Multi-scale SG: each window applied to MSC-corrected spectrum
print(f"=== Experiment {EXP}: Multi-scale SG (w={WINDOWS}) + Joint EPO ===")
print(f"Base: P1 (LOSO={P1_BASELINE}, LB=15.395)\n")
print("Building multi-scale features...")

Xtr_blocks = []
Xte_blocks = []
for w in WINDOWS:
    Xtr_w = sg_deriv(Xtr_msc, window=w, polyorder=2)
    Xte_w = sg_deriv(Xte_msc, window=w, polyorder=2)
    Xtr_blocks.append(Xtr_w)
    Xte_blocks.append(Xte_w)
    print(f"  w={w:2d}: shape={Xtr_w.shape}")

Xtr_concat = np.hstack(Xtr_blocks)
Xte_concat = np.hstack(Xte_blocks)
print(f"  concat: {Xtr_concat.shape}  (expected: {len(X_train_raw)} × {1555 * len(WINDOWS)})")

# Joint EPO on 4665-dim concatenated space
print("\nFitting Joint EPO (n=5) on concatenated space...")
V = compute_epo_matrix(Xtr_concat, y_train, sp_train, n_components=5)
print(f"  EPO matrix shape: {V.shape}")

Xtr = apply_epo(Xtr_concat, V)
Xte = apply_epo(Xte_concat, V)
print(f"  After EPO: train={Xtr.shape}, test={Xte.shape}")

# ── LGBM params (P1-base, ff adjusted for 4665 features) ─────────────────────
params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02,
          "num_leaves": 63,
          "feature_fraction": FF,
          "min_child_samples": 10}

print(f"\nff={FF} → ~{int(Xtr.shape[1] * FF)} features/tree "
      f"(P1 equiv: {int(1555 * 0.07)})\n")

# ── LOSO-CV with y^0.27 transform ─────────────────────────────────────────────
y_p027 = y_train ** 0.27
inv    = lambda pred: np.clip(pred, 0, None) ** (1.0 / 0.27)

oof_trans = np.zeros(len(y_train))
iters = []

print(f"{'fold':>6}  {'sp':>4}  {'n_val':>6}  {'best_iter':>10}")
print("-" * 34)

for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xtr[va_idx])
    iters.append(m.best_iteration)
    fold_rmse = loso_rmse(inv(oof_trans[va_idx]), y_train[va_idx])
    print(f"  sp{sp:2d}  {len(va_idx):6d}  {m.best_iteration:10d}  fold_rmse={fold_rmse:.4f}")

oof     = inv(oof_trans)
rmse    = loso_rmse(oof, y_train)
avg_iter = int(np.mean(iters))
diff    = rmse - P1_BASELINE

print(f"\n{'='*50}")
print(f"T1-lite LOSO-RMSE : {rmse:.4f}")
print(f"P1 baseline       : {P1_BASELINE:.4f}")
print(f"Delta             : {diff:+.4f}  {'✅ 改善' if diff < 0 else '❌ 悪化'}")
print(f"avg_iter          : {avg_iter}")
print(f"ff                : {FF} (~{int(Xtr.shape[1]*FF)} feat/tree)")

# ── Per-species breakdown ──────────────────────────────────────────────────────
print("\nPer-species RMSE:")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    sp_rmse = loso_rmse(oof[idx], y_train[idx])
    print(f"  sp{sp:2d}: n={len(idx):3d}  RMSE={sp_rmse:.4f}")

# ── Final model & submission ───────────────────────────────────────────────────
if rmse < P1_BASELINE:
    print(f"\n✅ P1を超えた → 最終モデル学習・提出実行")
    dtrain_f = lgb.Dataset(Xtr, label=y_p027)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=avg_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte))
    OUT   = f"{OUT_DIR}/submission_{EXP}_multiscale_w{'-'.join(str(w) for w in WINDOWS)}.csv"
    save_submission(test_ids, preds, OUT)
    memo  = f"{EXP}: 3-scale SG w={WINDOWS} Joint-EPO ff={FF} LOSO={rmse:.4f}"
    submit_to_signate(OUT, memo, loso=rmse)
else:
    print(f"\n❌ P1({P1_BASELINE})を超えず → 提出スキップ")
    print(f"   手動検討: マルチスケールで情報増分なし。w幅変更・poly変更・4-scale検討")
