"""
Experiment L1: I2パイプライン専用LGBMハイパラ再チューニング
==========================================================
I2のパイプライン (MSC+SG(w=9,p=2)+EPO(n=5)+sqrt(y)) に対して
LGBMのnum_leaves / min_child_samples を再探索する。

現在のI2パラメータ: lr=0.02, leaves=63, ff=0.07, mcs=10 (T実験流用)
→ SG(w=9,p=2)+EPO後の特徴空間に最適化されていない可能性。

グリッド: leaves x mcs の総当たり
  num_leaves:       [31, 63, 127]
  min_child_samples:[5, 10, 20, 30]
  (lr=0.02, ff=0.07 は固定)

ベース: I2 (LOSO=15.73, LB=16.101)
"""
import sys
import numpy as np
from itertools import product
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "L1"

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

# ── データ読み込み & I2固定前処理 ─────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]
y_sqrt = np.sqrt(y_train)

ref = X_train_raw.mean(axis=0)
Xtr_pp = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_pp = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V = compute_epo_matrix(Xtr_pp, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_pp, V)
Xte = apply_epo(Xte_pp, V)

# ── グリッドサーチ ─────────────────────────────────────────────────────────────
leaves_grid = [31, 63, 127]
mcs_grid    = [5, 10, 20, 30]

print(f"=== Experiment {EXP}: I2パイプライン LGBMハイパラ再チューニング ===")
print(f"固定: lr=0.02, ff=0.07 | 探索: num_leaves x min_child_samples")
print(f"I2ベース(leaves=63, mcs=10): LOSO=15.73\n")
print(f"{'leaves':>7}  {'mcs':>4}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 38)

best_rmse = np.inf; best_cfg = None; best_bi = None

for leaves, mcs in product(leaves_grid, mcs_grid):
    params = {
        **LGBM_BASE_PARAMS,
        "learning_rate": 0.02,
        "num_leaves": leaves,
        "feature_fraction": 0.07,
        "min_child_samples": mcs,
    }

    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)

    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    i2_tag = " [I2]" if (leaves, mcs) == (63, 10) else ""
    flag   = " <-- best" if rmse < best_rmse else ""
    print(f"  {leaves:5d}  {mcs:4d}  {rmse:8.4f}  {avg_r:9d}{i2_tag}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = (leaves, mcs); best_bi = best_iters

bl, bm = best_cfg
print(f"\nBest: leaves={bl}, mcs={bm}  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

# ── ベスト設定で提出ファイル生成 ──────────────────────────────────────────────
best_params = {
    **LGBM_BASE_PARAMS,
    "learning_rate": 0.02,
    "num_leaves": bl,
    "feature_fraction": 0.07,
    "min_child_samples": bm,
}
dtrain_f = lgb.Dataset(Xtr, label=y_sqrt)
final = lgb.train(best_params, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte), 0, None) ** 2

OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_L1_i2_l{bl}_m{bm}.csv"
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, f"L1: I2+LGBM tune leaves={bl} mcs={bm} LOSO={best_rmse:.4f}", loso=best_rmse)
print(f"\n[Done] {OUT}")
