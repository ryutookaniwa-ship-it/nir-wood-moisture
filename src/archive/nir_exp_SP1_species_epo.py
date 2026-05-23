"""
Experiment SP1: テスト種別 EPO アンサンブル
============================================
各テスト種(2,6,7,9,10,18)に対し、その種の疑似ラベルを含む専用EPOを計算し
その種サンプルの予測に使用する。

テスト種 T_i の予測:
  EPO_i = compute_epo(train_13種 + T_i, pseudo_Ti)
  LGBM trained on apply_epo(Xtr_sg, EPO_i)
  predict apply_epo(Xte_sg, EPO_i)[sp_test == T_i]

LOSO: 6モデルのOOF平均で評価
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

EXP = "SP1"
P_POWER = 0.27
N_ROUNDS_P1  = 600
N_ROUNDS_PL  = 616
TEST_SPECIES = [2, 6, 7, 9, 10, 18]

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

# ── 疑似ラベル v1 生成 ─────────────────────────────────────────────────────
print("疑似ラベル v1 生成...")
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
pseudo_v1_all = np.clip(m1.predict(Xte_epo), 0, None) ** (1 / P_POWER)
print(f"  v1: mean={pseudo_v1_all.mean():.2f}")

# ── テスト種別 EPO ────────────────────────────────────────────────────────
print(f"\n=== {EXP}: 6種別EPO アンサンブル ===")
oof_stack = np.zeros((len(y_train), len(TEST_SPECIES)))
te_preds  = np.zeros((len(sp_test),  len(TEST_SPECIES)))

for i, ts in enumerate(TEST_SPECIES):
    mask_ts   = sp_test == ts
    Xte_ts_sg = Xte_sg[mask_ts]
    pseudo_ts = pseudo_v1_all[mask_ts]

    X_comb  = np.vstack([Xtr_sg, Xte_ts_sg])
    y_comb  = np.concatenate([y_train, pseudo_ts])
    sp_comb = np.concatenate([sp_train, np.full(mask_ts.sum(), ts)])
    V_ts    = compute_epo_matrix(X_comb, y_comb, sp_comb, n_components=5)

    Xtr_ts = apply_epo(Xtr_sg, V_ts)
    Xte_ts = apply_epo(Xte_sg, V_ts)

    iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_ts[tr_idx], label=y_train[tr_idx] ** P_POWER)
        dval   = lgb.Dataset(Xtr_ts[va_idx], label=y_train[va_idx] ** P_POWER, reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_stack[va_idx, i] = np.clip(m.predict(Xtr_ts[va_idx]), 0, None) ** (1 / P_POWER)
        iters.append(m.best_iteration)

    avg_r = int(np.mean(iters))
    loso_i = loso_rmse(oof_stack[:, i], y_train)
    print(f"  sp{ts:2d}: LOSO={loso_i:.4f}  avg_iter={avg_r}")

    mf = lgb.train(P1_PARAMS, lgb.Dataset(Xtr_ts, label=y_train ** P_POWER),
                   num_boost_round=avg_r, callbacks=[lgb.log_evaluation(-1)])
    te_preds[:, i] = np.clip(mf.predict(Xte_ts), 0, None) ** (1 / P_POWER)

# ── アンサンブル評価 ─────────────────────────────────────────────────────
oof_avg  = oof_stack.mean(axis=1)
rmse_avg = loso_rmse(oof_avg, y_train)
print(f"\n6モデル平均 LOSO: {rmse_avg:.4f}  (P1=15.4725, PL2=15.0623, delta={rmse_avg-15.4725:+.4f})")

# テスト予測: 各種にその種専用EPOモデルの予測を使用
preds = np.zeros(len(sp_test))
for i, ts in enumerate(TEST_SPECIES):
    preds[sp_test == ts] = te_preds[sp_test == ts, i]

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: sp_EPO_ensemble, LOSO={rmse_avg:.4f}", loso=rmse_avg)
print(f"[Done] {EXP}")
