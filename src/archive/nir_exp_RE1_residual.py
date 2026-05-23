"""
Experiment RE1: 残差アンサンブル (P1ベース + delta特徴量で残差補正)
==================================================================
設計:
  Stage1: MSC+SG+EPO → LGBM(P1) → 粗い予測
  Stage2: delta_prev(3列) → Ridge → 残差補正
  Final:  Stage1 + Stage2

LOSO-CV手順:
  1. 全体LOSO で Stage1 OOF を計算
  2. 残差 resid = y_train - Stage1_OOF
  3. 各foldで:
     - Stage2: training種のdelta特徴量でRidge学習 (残差ターゲット)
     - validation種の残差を予測
  4. final_oof = Stage1_OOF + Stage2_OOF

直感: Stage1が絶対的なMCレベルを捉え、Stage2が乾燥ダイナミクスの
      誤差を補正する。異なる前処理→相補性が期待できる。
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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

EXP = "RE1"
TARGET_WNS = (5200.0, 7000.0)
P_POWER = 0.27
RIDGE_ALPHA = 1.0

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
    return delta


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

# ── P1前処理 ──────────────────────────────────────────────────────────────────
ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── delta特徴量 ──────────────────────────────────────────────────────────────
delta_tr = compute_delta_features(Xtr_sg, train_ids, sp_train, wns)
delta_te = compute_delta_features(Xte_sg, test_ids,  sp_test,  wns)

# ── Stage1: P1 LOSO OOF ───────────────────────────────────────────────────────
print("=== Stage1: P1 LOSO OOF ===")
oof_s1 = np.zeros(len(y_train)); best_iters_s1 = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_pow[tr_idx])
    dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_pow[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_s1[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** (1 / P_POWER)
    best_iters_s1.append(m.best_iteration)

rmse_s1 = loso_rmse(oof_s1, y_train)
print(f"Stage1 LOSO-RMSE: {rmse_s1:.4f}")

# ── Stage2: delta特徴量でLOSO残差補正 ─────────────────────────────────────────
print("\n=== Stage2: Ridge on delta_prev (residual correction) ===")
resid_global = y_train - oof_s1   # 全体の残差

oof_s2 = np.zeros(len(y_train))
for tr_idx, va_idx, sp in loso_folds(sp_train):
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(delta_tr[tr_idx], resid_global[tr_idx])
    oof_s2[va_idx] = ridge.predict(delta_tr[va_idx])

# ── 最終スコア ────────────────────────────────────────────────────────────────
oof_final = oof_s1 + oof_s2
rmse_final = loso_rmse(oof_final, y_train)

print(f"\n=== {EXP}: 結果サマリ ===")
print(f"Stage1 LOSO-RMSE : {rmse_s1:.4f}")
print(f"Final  LOSO-RMSE : {rmse_final:.4f}  (P1=15.4725, delta={rmse_final-15.4725:+.4f})")
print(f"Stage2 残差補正  : {rmse_final-rmse_s1:+.4f}")

# ── 最終モデル構築 (全データ) ──────────────────────────────────────────────────
avg_r = int(np.mean(best_iters_s1))
dtrain_f = lgb.Dataset(Xtr_epo, label=y_pow)
final_s1 = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
                     callbacks=[lgb.log_evaluation(-1)])
preds_s1 = np.clip(final_s1.predict(Xte_epo), 0, None) ** (1 / P_POWER)

# Stage2: 全訓練残差で再学習
final_s2 = Ridge(alpha=RIDGE_ALPHA)
final_s2.fit(delta_tr, resid_global)
preds_s2 = final_s2.predict(delta_te)

preds_final = np.clip(preds_s1 + preds_s2, 0, None)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds_final, OUT)
submit_to_signate(OUT, memo=f"{EXP}: P1+Ridge_residual, LOSO={rmse_final:.4f}", loso=rmse_final)
print(f"[Done] {EXP}")
