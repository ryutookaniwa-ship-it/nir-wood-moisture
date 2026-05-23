"""
Experiment W2: train類似度k-NN特徴量
======================================
根拠: 競技ルール明示的OK（主催者確認済み）
     「学習済みデータを基準に新規データを評価する」行為は実運用に相当。
     C2（樹種レベルクラスタリング）とは異なりサンプルレベルk-NNは未試行。

実装:
  - MSC後スペクトル(SG微分前)でcosine類似度を計算
  - LOSO-CV: 検証foldはspecies-based LOSO-consistent k-NN特徴量を使用
             訓練foldはleave-one-out k-NN特徴量を使用(事前計算)
  - test k-NN: 全trainを参照セット

追加特徴量 (k=5, 10, 20):
  knn_mc_k5    : top-5近傍の類似度加重平均MC
  knn_mc_k10   : top-10近傍の類似度加重平均MC
  knn_mc_k20   : top-20近傍の類似度加重平均MC
  sim_max       : 最大cosine類似度
  sim_top5_mean : top-5類似度の平均

ベース: P1 (LOSO=15.4725, LB=15.395)
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")

from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP      = "W2"
OUT_DIR  = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_LOSO  = 15.4725
K_VALUES = [5, 10, 20]

# ── EPO ──────────────────────────────────────────────────────────────────────
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

def _knn_feats_from_sim(sim, y_ref, k_values):
    """sim: (n_query, n_ref)行列から k-NN 特徴量を生成。"""
    feats = []
    for k in k_values:
        k_eff   = min(k, sim.shape[1])
        top_idx = np.argsort(sim, axis=1)[:, -k_eff:]
        top_sim = np.take_along_axis(sim, top_idx, axis=1)
        top_y   = y_ref[top_idx]
        w       = top_sim / (top_sim.sum(axis=1, keepdims=True) + 1e-10)
        feats.append((top_y * w).sum(axis=1))

    sim_max      = sim.max(axis=1)
    top5_idx     = np.argsort(sim, axis=1)[:, -5:]
    sim_top5mean = np.take_along_axis(sim, top5_idx, axis=1).mean(axis=1)
    feats.extend([sim_max, sim_top5mean])
    return np.column_stack(feats)

# ── Load & preprocess ─────────────────────────────────────────────────────────
data     = load_data()
y_train  = data["y_train"]
Xtr_raw  = data["X_train_raw"]
Xte_raw  = data["X_test_raw"]
test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref      = Xtr_raw.mean(axis=0)
Xtr_msc  = msc(Xtr_raw, ref)     # MSC only — 類似度計算用
Xte_msc  = msc(Xte_raw, ref)

Xtr_sg   = sg_deriv(Xtr_msc, window=9, polyorder=2)
Xte_sg   = sg_deriv(Xte_msc, window=9, polyorder=2)
V        = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo  = apply_epo(Xtr_sg, V)
Xte_epo  = apply_epo(Xte_sg, V)

n_knn    = len(K_VALUES) + 2
print(f"=== Experiment {EXP}: train k-NN 類似度特徴量 ===")
print(f"EPO: {Xtr_epo.shape[1]}次元 + k-NN: {n_knn}次元 = {Xtr_epo.shape[1]+n_knn}次元")

# ── 1. LOSO-consistent val-fold k-NN OOF ─────────────────────────────────────
# val fold の k-NN 参照 = training fold のみ（ルール準拠・リーク防止）
print("\n--- LOSO-consistent val OOF k-NN 計算 ---")
oof_knn = np.zeros((len(y_train), n_knn))
for tr_idx, va_idx, sp in loso_folds(sp_train):
    sim_va = cosine_similarity(Xtr_msc[va_idx], Xtr_msc[tr_idx])
    oof_knn[va_idx] = _knn_feats_from_sim(sim_va, y_train[tr_idx], K_VALUES)
    print(f"  sp{sp:02d}: n={len(va_idx):3d}  avg_sim_max={oof_knn[va_idx,-2].mean():.4f}  "
          f"avg_knn_mc_k5={oof_knn[va_idx,0].mean():.1f}%")

# ── 2. Leave-one-out k-NN for training samples ───────────────────────────────
# 訓練fold全体に使用。(n_train × n_train) cosine sim 行列で一括計算
print("\n--- Leave-one-out k-NN for training samples (全1322×1322)... ---")
sim_all = cosine_similarity(Xtr_msc)         # (1322, 1322)
np.fill_diagonal(sim_all, -np.inf)            # self 除外
knn_loo = _knn_feats_from_sim(sim_all, y_train, K_VALUES)
print(f"  LOO k-NN完了: avg_sim_max={knn_loo[:,-2].mean():.4f}  "
      f"avg_knn_mc_k5={knn_loo[:,0].mean():.1f}%")

# ── 3. Test k-NN: 全train参照 ────────────────────────────────────────────────
sim_te  = cosine_similarity(Xte_msc, Xtr_msc)
test_knn = _knn_feats_from_sim(sim_te, y_train, K_VALUES)
print(f"  Test k-NN: avg_sim_max={test_knn[:,-2].mean():.4f}  "
      f"avg_knn_mc_k5={test_knn[:,0].mean():.1f}%")

# ── Correlation analysis ──────────────────────────────────────────────────────
feat_names = [f"knn_mc_k{k}" for k in K_VALUES] + ["sim_max", "sim_top5_mean"]
print("\nCorr(k-NN feature, y_train):")
for i, name in enumerate(feat_names):
    r = np.corrcoef(oof_knn[:, i], y_train)[0, 1]
    print(f"  {name:18s}: {r:+.4f}")

# ── 4. LOSO-CV ───────────────────────────────────────────────────────────────
params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}
p = 0.27

print(f"\n--- LOSO-CV (y^{p:.2f}) ---")
y_trans   = y_train ** p
oof_trans = np.zeros(len(y_train))
iters     = []

for tr_idx, va_idx, sp in loso_folds(sp_train):
    # train fold: EPO + LOO k-NN    val fold: EPO + LOSO-consistent k-NN OOF
    Xtr_fold = np.hstack([Xtr_epo[tr_idx], knn_loo[tr_idx]])
    Xva_fold = np.hstack([Xtr_epo[va_idx], oof_knn[va_idx]])

    dtrain = lgb.Dataset(Xtr_fold, label=y_trans[tr_idx])
    dval   = lgb.Dataset(Xva_fold, label=y_trans[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xva_fold)
    iters.append(m.best_iteration)

oof      = np.clip(oof_trans, 0, None) ** (1.0 / p)
rmse     = loso_rmse(oof, y_train)
avg_iter = int(np.mean(iters))
delta    = rmse - P1_LOSO

print(f"\nW2  LOSO={rmse:.4f}  avg_iter={avg_iter}  vs P1: {delta:+.4f}")

print("\nPer-species RMSE:")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    sp_rmse = np.sqrt(np.mean((y_train[idx] - oof[idx])**2))
    print(f"  sp{sp:02d}: {sp_rmse:6.2f}")

# ── Submit if improved ────────────────────────────────────────────────────────
if rmse < P1_LOSO:
    print(f"\n✅ P1 より {-delta:.4f} 改善 → 提出")
    # Final model: LOO k-NN for train, all-train k-NN for test
    Xtr_final = np.hstack([Xtr_epo, knn_loo])
    y_full    = y_train ** p
    dtrain_f  = lgb.Dataset(Xtr_final, label=y_full)
    final     = lgb.train(params, dtrain_f,
                          num_boost_round=avg_iter,
                          callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte), 0, None) ** (1.0 / p)
    OUT = f"{OUT_DIR}/submission_{EXP}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: +kNN_sim LOSO={rmse:.4f}", loso=rmse)
else:
    print(f"\n❌ P1比 {delta:+.4f} 悪化 → 提出なし")
