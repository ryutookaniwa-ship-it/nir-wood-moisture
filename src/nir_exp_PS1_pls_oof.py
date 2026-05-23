"""
Experiment PS1: PLS OOF予測 (1列) を追加特徴量としてP1に結合
==============================================================
アイデア: PLS(n=8)のLOSO OOF予測をスタッキング特徴量として使用。
  - 実験K (PLS成分→LGBM): LOSO=36.19 で失敗
  - PS1 は「PLSのスカラー予測値1列」をLGBMへの補助特徴として渡す
    → 全く異なるアプローチ

実装:
  訓練: LOSO-CVでPLS OOF予測を生成 → EPO特徴量に結合
  テスト: 全訓練データで学習したPLSの予測値 → EPO特徴量に結合
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

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

EXP = "PS1"
PLS_N = 8
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


# ── Load ───────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")

target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
wns        = np.array([float(c) for c in spec_cols])
y_train    = train[target_col].values
X_tr_raw   = train[spec_cols].values.astype(np.float64)
X_te_raw   = test[spec_cols].values.astype(np.float64)
test_ids   = test["sample number"].values
sp_train   = train["species number"].values
y_pow      = y_train ** P_POWER

# ── P1前処理 ──────────────────────────────────────────────────────────────────
ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── PLS OOF予測 (訓練) ────────────────────────────────────────────────────────
print("Computing PLS LOSO OOF predictions...")
pls_oof = np.zeros(len(y_train))
for tr_idx, va_idx, sp in loso_folds(sp_train):
    pls = PLSRegression(n_components=PLS_N, scale=True)
    pls.fit(Xtr_sg[tr_idx], y_train[tr_idx])
    pls_oof[va_idx] = pls.predict(Xtr_sg[va_idx]).ravel()

print(f"PLS OOF RMSE: {loso_rmse(pls_oof, y_train):.4f}")

# ── PLS全データ学習 (テスト用) ─────────────────────────────────────────────────
pls_full = PLSRegression(n_components=PLS_N, scale=True)
pls_full.fit(Xtr_sg, y_train)
pls_test = pls_full.predict(Xte_sg).ravel()

# ── 結合 ─────────────────────────────────────────────────────────────────────
Xtr_full = np.hstack([Xtr_epo, pls_oof.reshape(-1, 1)])   # (1322, 1556)
Xte_full = np.hstack([Xte_epo, pls_test.reshape(-1, 1)])
print(f"Feature shape: {Xtr_full.shape}  (EPO 1555 + PLS_OOF 1)")

# ── LOSO-CV ───────────────────────────────────────────────────────────────────
print(f"\n=== {EXP}: PLS OOF(1) + P1 ===")
oof = np.zeros(len(y_train)); best_iters = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_full[tr_idx], label=y_pow[tr_idx])
    dval   = lgb.Dataset(Xtr_full[va_idx], label=y_pow[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(Xtr_full[va_idx]), 0, None) ** (1 / P_POWER)
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
print(f"LOSO-RMSE: {rmse:.4f}  (P1=15.4725, delta={rmse-15.4725:+.4f})")
print(f"avg_iter : {avg_r}")

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
dtrain_f = lgb.Dataset(Xtr_full, label=y_pow)
final = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_full), 0, None) ** (1 / P_POWER)
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: PLS_OOF+P1, LOSO={rmse:.4f}", loso=rmse)
print(f"[Done] {EXP}")
