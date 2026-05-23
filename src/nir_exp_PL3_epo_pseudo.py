"""
Experiment PL3: 疑似ラベルで EPO 再計算 (19種 EPO)
===================================================
PL2 の疑似ラベル v1 を使い、train(13種)+test(6種) の
全19種でEPO行列を再推定。より精度の高い樹種固有方向除去を目指す。
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

EXP = "PL3"
P_POWER = 0.27
N_ROUNDS_P1 = 600
N_ROUNDS_PL = 616

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


train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
y_train  = train[target_col].values
X_tr_raw = train[spec_cols].values.astype(np.float64)
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_train = train["species number"].values
sp_test  = test["species number"].values

ref    = X_tr_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)

# ── Step1: P1 EPO(13種) → 疑似ラベル v1 ─────────────────────────────────
print("Step1: 疑似ラベル v1 生成 (13種 EPO)...")
V13     = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V13)
Xte_epo = apply_epo(Xte_sg, V13)

m0 = lgb.train(P1_PARAMS, lgb.Dataset(Xtr_epo, label=y_train ** P_POWER),
               num_boost_round=N_ROUNDS_P1, callbacks=[lgb.log_evaluation(-1)])
pseudo_v0 = np.clip(m0.predict(Xte_epo), 0, None) ** (1 / P_POWER)

m1 = lgb.train(P1_PARAMS,
               lgb.Dataset(np.vstack([Xtr_epo, Xte_epo]),
                           label=np.concatenate([y_train, pseudo_v0]) ** P_POWER),
               num_boost_round=N_ROUNDS_PL, callbacks=[lgb.log_evaluation(-1)])
pseudo_v1 = np.clip(m1.predict(Xte_epo), 0, None) ** (1 / P_POWER)
print(f"  v1: mean={pseudo_v1.mean():.2f}, std={pseudo_v1.std():.2f}")

# ── Step2: 19種 EPO 再計算 ────────────────────────────────────────────────
print("\nStep2: 19種 EPO 計算 (train+test)...")
X_sg_all = np.vstack([Xtr_sg, Xte_sg])
y_all    = np.concatenate([y_train, pseudo_v1])
sp_all   = np.concatenate([sp_train, sp_test])
V19      = compute_epo_matrix(X_sg_all, y_all, sp_all, n_components=5)
Xtr_epo19 = apply_epo(Xtr_sg, V19)
Xte_epo19 = apply_epo(Xte_sg, V19)
print(f"  V13 shape={V13.shape}, V19 shape={V19.shape}")

# ── Step3: 標準 LOSO-CV ──────────────────────────────────────────────────
print("\nStep3: LOSO-CV (19種 EPO)...")
oof = np.zeros(len(y_train)); iters = []
for tr_idx, va_idx, _ in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_epo19[tr_idx], label=y_train[tr_idx] ** P_POWER)
    dval   = lgb.Dataset(Xtr_epo19[va_idx], label=y_train[va_idx] ** P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(Xtr_epo19[va_idx]), 0, None) ** (1 / P_POWER)
    iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train)
avg_r = int(np.mean(iters))
print(f"PL3 LOSO: {rmse:.4f}  (P1=15.4725, PL2=15.0623, delta={rmse-15.4725:+.4f})")
print(f"avg_iter: {avg_r}")

# ── 提出 ─────────────────────────────────────────────────────────────────
dtrain_f = lgb.Dataset(Xtr_epo19, label=y_train ** P_POWER)
mf = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
               callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(mf.predict(Xte_epo19), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: 19sp_EPO, LOSO={rmse:.4f}", loso=rmse)
print(f"[Done] {EXP}")
