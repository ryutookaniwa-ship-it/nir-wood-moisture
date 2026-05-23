"""
Experiment GS2: delta_prev特徴量のみ (3列) + P1
================================================
GS1 (+0.58悪化) の原因分析:
  position_ratio が樹種ごとに意味するMCが異なる → 混乱
  delta_prev は「スペクトル変化の方向」= 樹種を超えて普遍的な可能性

GS2: position_index/ratio/rolling を除き、delta_prev 3列のみ追加
  - delta_prev_5200  : 前回測定から5200cm⁻¹吸収の変化
  - delta_prev_7000  : 前回測定から7000cm⁻¹吸収の変化
  - delta_prev_mean  : 前回測定から全波長平均の変化
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

EXP = "GS2"
TARGET_WNS = (5200.0, 7000.0)
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


def compute_delta_features(X_sg, sample_ids, species, wns, target_wns=TARGET_WNS):
    """前回測定からのスペクトル変化量 (3列) を計算。最初の測定は0埋め。"""
    n_samples = X_sg.shape[0]
    target_indices = [int(np.argmin(np.abs(wns - t))) for t in target_wns]
    series = np.column_stack(
        [X_sg[:, idx] for idx in target_indices] + [np.mean(X_sg, axis=1)]
    ).astype(np.float32)

    delta = np.zeros((n_samples, series.shape[1]), dtype=np.float32)

    for sp in np.unique(species):
        grp_idx = np.where(species == sp)[0]
        order = np.argsort(sample_ids[grp_idx], kind="stable")
        ordered = grp_idx[order]
        for pos in range(1, len(ordered)):
            delta[ordered[pos]] = series[ordered[pos]] - series[ordered[pos - 1]]

    return delta  # (n_samples, 3)


# ── Load ───────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")

target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
wns        = np.array([float(c) for c in spec_cols])
y_train    = train[target_col].values
X_tr_raw   = train[spec_cols].values.astype(np.float64)
X_te_raw   = test[spec_cols].values.astype(np.float64)
train_ids  = train["sample number"].values
test_ids   = test["sample number"].values
sp_train   = train["species number"].values
sp_test    = test["species number"].values
y_pow      = y_train ** P_POWER

ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

delta_tr = compute_delta_features(Xtr_sg, train_ids, sp_train, wns)
delta_te = compute_delta_features(Xte_sg, test_ids,  sp_test,  wns)

Xtr_full = np.hstack([Xtr_epo, delta_tr])  # (1322, 1558)
Xte_full = np.hstack([Xte_epo, delta_te])
print(f"Feature shape: {Xtr_full.shape}  (EPO 1555 + delta 3)")

# ── LOSO-CV ───────────────────────────────────────────────────────────────────
print(f"\n=== {EXP}: delta_prev(3) + P1 ===")
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
submit_to_signate(OUT, memo=f"{EXP}: delta_prev+P1, LOSO={rmse:.4f}", loso=rmse)
print(f"[Done] {EXP}")
