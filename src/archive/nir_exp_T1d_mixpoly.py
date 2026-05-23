"""
Experiment T1d: 混合poly SG concat + Joint EPO (別アプローチ)
=============================================================
仮説:
  実験履歴より:
    w=5 + poly=3 → LOSO=19.55 (実験T, EPO前ベスト)
    w=9 + poly=2 → LOSO=15.73 (実験I2, EPO後ベスト)
  poly=3はw=5専用スイートスポット(J4で確認)。
  両者をconcatすることで、poly=3の高周波鋭角情報とpoly=2の中周波情報を融合。

設計:
  [MSC→SG(w=5, poly=3)] + [MSC→SG(w=9, poly=2)] → concat (3110-dim)
  → Joint EPO(n=5) → LGBM(ff=0.035)

  各スケールが異なるpoly=最適値を持つという物理的根拠に基づく。

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

EXP = "T1d"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_BASELINE = 15.4725

# 混合poly設定
SCALE_CONFIGS = [(5, 3), (9, 2)]   # (window, polyorder)
N_FEAT_EXPECTED = 1555 * len(SCALE_CONFIGS)  # 3110
FF = round(108 / N_FEAT_EXPECTED, 4)         # ~0.0347


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

ref = X_train_raw.mean(axis=0)
Xtr_msc = msc(X_train_raw, ref)
Xte_msc = msc(X_test_raw, ref)

print(f"=== Experiment {EXP}: Mixed-poly SG concat + Joint EPO ===")
print(f"Scale configs: {SCALE_CONFIGS}  (w, poly)")
print(f"ff={FF} -> ~{int(N_FEAT_EXPECTED*FF)} feat/tree\n")

Xtr_blocks = []; Xte_blocks = []
for (w, p) in SCALE_CONFIGS:
    Xtr_w = sg_deriv(Xtr_msc, window=w, polyorder=p)
    Xte_w = sg_deriv(Xte_msc, window=w, polyorder=p)
    Xtr_blocks.append(Xtr_w); Xte_blocks.append(Xte_w)
    print(f"  w={w}, poly={p}: shape={Xtr_w.shape}")

Xtr_concat = np.hstack(Xtr_blocks)
Xte_concat = np.hstack(Xte_blocks)
print(f"  concat: {Xtr_concat.shape}")

V = compute_epo_matrix(Xtr_concat, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_concat, V)
Xte = apply_epo(Xte_concat, V)

y_p027 = y_train ** 0.27
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / 0.27)

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": FF, "min_child_samples": 10}

print(f"\nBase: P1(LOSO={P1_BASELINE})\n")
print(f"{'fold':>6}  {'sp':>4}  {'n_val':>6}  {'best_iter':>10}  {'fold_rmse':>10}")
print("-" * 46)

oof_trans = np.zeros(len(y_train)); iters = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xtr[va_idx])
    iters.append(m.best_iteration)
    fold_rmse = loso_rmse(inv(oof_trans[va_idx]), y_train[va_idx])
    print(f"  sp{sp:2d}  {len(va_idx):6d}  {m.best_iteration:10d}  {fold_rmse:10.4f}")

oof = inv(oof_trans); rmse = loso_rmse(oof, y_train); avg_iter = int(np.mean(iters))
diff = rmse - P1_BASELINE

print(f"\n{'='*50}")
print(f"{EXP} LOSO-RMSE : {rmse:.4f}")
print(f"P1 baseline  : {P1_BASELINE:.4f}")
print(f"Delta        : {diff:+.4f}  ({'IMPROVED' if diff < 0 else 'worse'})")
print(f"avg_iter     : {avg_iter}")

print("\nPer-species RMSE:")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    print(f"  sp{sp:2d}: n={len(idx):3d}  RMSE={loso_rmse(oof[idx], y_train[idx]):.4f}")

if rmse < P1_BASELINE:
    dtrain_f = lgb.Dataset(Xtr, label=y_p027)
    final = lgb.train(params, dtrain_f, num_boost_round=avg_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte))
    OUT = f"{OUT_DIR}/submission_{EXP}_mixpoly_w5p3_w9p2.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: mixpoly (w5p3+w9p2) Joint-EPO ff={FF} LOSO={rmse:.4f}", loso=rmse)
else:
    print("\nP1 baseline not beaten -> skip submission")
    print("   混合polyの情報増分なし。単一スケール(P1)が依然最適。")
