"""
Experiment CO1: CORAL (Correlation Alignment) ドメイン適応
===========================================================
EPO は訓練/テスト間の「分散方向(1次)」を除去。
CORAL は「共分散構造(2次)」を揃える。

簡易CORAL (特徴量ごとの平均・標準偏差整合):
  X_train_aligned[:, i] = (X_train_epo[:, i] - mu_train[i]) / sigma_train[i]
                           * sigma_test[i] + mu_test[i]
  X_test: 変更なし

直感: EPO後の訓練特徴量をテスト分布に合わせてスケール変換することで
      訓練-テスト間の残余ギャップを縮小する。

注意: sigma=0 の列は変換スキップ (ゼロ除算防止)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP = "CO1"
P_POWER = 0.27

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
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


def coral_align(X_src, X_tgt):
    """
    特徴量ごとの平均・標準偏差をソース→ターゲット分布に整合。
    X_src: 訓練EPO特徴量 (n_train, n_feat)
    X_tgt: テストEPO特徴量 (n_test, n_feat)
    戻り値: X_src_aligned (n_train, n_feat)
    """
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    std_src = X_src.std(axis=0)
    std_tgt = X_tgt.std(axis=0)

    # std=0の列はスキップ (変換なし)
    safe_mask = std_src > 1e-10
    X_aligned = X_src.copy()
    X_aligned[:, safe_mask] = (
        (X_src[:, safe_mask] - mu_src[safe_mask]) / std_src[safe_mask]
        * std_tgt[safe_mask] + mu_tgt[safe_mask]
    )
    return X_aligned


train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
y_train  = train[target_col].values
X_tr_raw = train[spec_cols].values.astype(np.float64)
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_train = train["species number"].values
y_pow    = y_train ** P_POWER

ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── CORAL整合 ────────────────────────────────────────────────────────────────
print("CORAL整合前後の統計:")
print(f"  Train EPO mean (before): {Xtr_epo.mean():.5f}")
print(f"  Test  EPO mean         : {Xte_epo.mean():.5f}")

Xtr_coral = coral_align(Xtr_epo, Xte_epo)

print(f"  Train EPO mean (after) : {Xtr_coral.mean():.5f}")
print(f"  Train EPO std  (before): {Xtr_epo.std():.5f}")
print(f"  Train EPO std  (after) : {Xtr_coral.std():.5f}")
print(f"  Test  EPO std          : {Xte_epo.std():.5f}")

# ── LOSO-CV (CORAL整合後) ──────────────────────────────────────────────────
print(f"\n=== {EXP}: CORAL + P1 ===")
oof = np.zeros(len(y_train)); best_iters = []
for tr_idx, va_idx, _ in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_coral[tr_idx], label=y_pow[tr_idx])
    dval   = lgb.Dataset(Xtr_coral[va_idx], label=y_pow[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(Xtr_coral[va_idx]), 0, None) ** (1 / P_POWER)
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train)
avg_r = int(np.mean(best_iters))
print(f"CO1 LOSO-RMSE: {rmse:.4f}  (P1=15.4725, delta={rmse-15.4725:+.4f})")
print(f"avg_iter     : {avg_r}")

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
dtrain_f = lgb.Dataset(Xtr_coral, label=y_pow)
final = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: CORAL, LOSO={rmse:.4f}", loso=rmse)
print(f"[Done] {EXP}")
