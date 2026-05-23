"""
Experiment EN1: P1 × PL2 OOF アンサンブル
==========================================
P1(LOSO=15.4725) と PL2(LOSO=15.0623) のOOF予測をブレンドし
最適 alpha を探索。残差相関が低ければ改善の余地あり。
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

EXP = "EN1"
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

ref     = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── P1 OOF ─────────────────────────────────────────────────────────────────
print("P1 OOF (標準LOSO)...")
oof_p1 = np.zeros(len(y_train))
for tr_idx, va_idx, _ in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_train[tr_idx] ** P_POWER)
    dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_train[va_idx] ** P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_p1[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** (1 / P_POWER)
print(f"  P1 LOSO={loso_rmse(oof_p1, y_train):.4f}")

# ── 疑似ラベル v1 生成 ─────────────────────────────────────────────────────
print("\n疑似ラベル v1 生成...")
m0 = lgb.train(P1_PARAMS, lgb.Dataset(Xtr_epo, label=y_train ** P_POWER),
               num_boost_round=N_ROUNDS_P1, callbacks=[lgb.log_evaluation(-1)])
pseudo_v0 = np.clip(m0.predict(Xte_epo), 0, None) ** (1 / P_POWER)

X_aug1 = np.vstack([Xtr_epo, Xte_epo])
y_aug1 = np.concatenate([y_train, pseudo_v0])
m1 = lgb.train(P1_PARAMS, lgb.Dataset(X_aug1, label=y_aug1 ** P_POWER),
               num_boost_round=N_ROUNDS_PL, callbacks=[lgb.log_evaluation(-1)])
pseudo_v1 = np.clip(m1.predict(Xte_epo), 0, None) ** (1 / P_POWER)

# ── PL2 OOF ────────────────────────────────────────────────────────────────
print("PL2 OOF (テストデータ込みLOSO)...")
X_aug2 = np.vstack([Xtr_epo, Xte_epo])
y_aug2 = np.concatenate([y_train, pseudo_v1])
te_idx = np.arange(len(y_train), len(y_aug2))

oof_pl2 = np.zeros(len(y_train)); iters = []
for tr_orig, va_orig, _ in loso_folds(sp_train):
    tr_aug = np.concatenate([tr_orig, te_idx])
    dtrain = lgb.Dataset(X_aug2[tr_aug], label=y_aug2[tr_aug] ** P_POWER)
    dval   = lgb.Dataset(X_aug2[va_orig], label=y_aug2[va_orig] ** P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_pl2[va_orig] = np.clip(m.predict(X_aug2[va_orig]), 0, None) ** (1 / P_POWER)
    iters.append(m.best_iteration)
avg_r_pl2 = int(np.mean(iters))
print(f"  PL2 LOSO={loso_rmse(oof_pl2, y_train):.4f}  avg_iter={avg_r_pl2}")

# ── 残差相関・alpha 探索 ───────────────────────────────────────────────────
corr = np.corrcoef(oof_p1, oof_pl2)[0, 1]
print(f"\n残差相関 r={corr:.4f}")
print(f"\n{'alpha':>6}  {'LOSO':>9}")
print("-" * 20)
best_rmse = np.inf; best_alpha = 1.0
for alpha in np.arange(0.0, 1.05, 0.1):
    rmse = loso_rmse(alpha * oof_pl2 + (1 - alpha) * oof_p1, y_train)
    tag = " <-- best" if rmse < best_rmse else ""
    print(f"  {alpha:.1f}   {rmse:9.4f}{tag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_alpha = round(alpha, 1)
print(f"\nbest alpha={best_alpha:.1f}, LOSO={best_rmse:.4f}")

# ── テスト予測 ─────────────────────────────────────────────────────────────
mp1 = lgb.train(P1_PARAMS, lgb.Dataset(Xtr_epo, label=y_train ** P_POWER),
                num_boost_round=N_ROUNDS_P1, callbacks=[lgb.log_evaluation(-1)])
preds_p1 = np.clip(mp1.predict(Xte_epo), 0, None) ** (1 / P_POWER)

mpl2 = lgb.train(P1_PARAMS, lgb.Dataset(X_aug2, label=y_aug2 ** P_POWER),
                 num_boost_round=avg_r_pl2, callbacks=[lgb.log_evaluation(-1)])
preds_pl2 = np.clip(mpl2.predict(Xte_epo), 0, None) ** (1 / P_POWER)

preds = best_alpha * preds_pl2 + (1 - best_alpha) * preds_p1

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: P1xPL2 a={best_alpha:.1f}, LOSO={best_rmse:.4f}", loso=best_rmse)
print(f"[Done] {EXP}")
