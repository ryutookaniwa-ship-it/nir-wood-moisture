"""
Experiment MH1: 複数種同時 Holdout CV でのハイパラ再探索
=========================================================
現LOSO(1種holdout)での既確定パラメータ:
  leaves=63, mcs=10, ff=0.07, lr=0.02

問題意識:
  テスト条件 = 「6種同時未知」なのに、LOSO = 「1種holdout」で最適化
  → より難しい「2〜3種同時holdout」で選ぶと別の最適値が出る可能性

MH1の設計:
  訓練13種を 6〜7 fold に分割 (各fold: 約2種を同時holdout)
  → ff / leaves を再探索し、best params を通常LOSOでも評価

Fold割り当て (species_number順に6グループ):
  Fold0: sp1, sp3        Fold1: sp4, sp5
  Fold2: sp8, sp11       Fold3: sp12, sp13
  Fold4: sp14, sp15      Fold5: sp16, sp17, sp19
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

EXP = "MH1"
P_POWER = 0.27

# 複数種holdout fold定義 (訓練13種を6グループに)
MULTI_FOLDS = [
    [1, 3],
    [4, 5],
    [8, 11],
    [12, 13],
    [14, 15],
    [16, 17, 19],
]


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


def multi_holdout_cv(X, y_pow, y_orig, sp, params, multi_folds, n_rounds=3000):
    """複数種同時holdout CVのRMSEを返す。"""
    oof = np.zeros(len(y_orig))
    best_iters = []
    for val_species in multi_folds:
        va_mask = np.isin(sp, val_species)
        tr_mask = ~va_mask
        va_idx = np.where(va_mask)[0]
        tr_idx = np.where(tr_mask)[0]
        dtrain = lgb.Dataset(X[tr_idx], label=y_pow[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=y_pow[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(X[va_idx]), 0, None) ** (1 / P_POWER)
        best_iters.append(m.best_iteration)
    return loso_rmse(oof, y_orig), int(np.mean(best_iters))


def loso_cv(X, y_pow, y_orig, sp, params, n_rounds=3000):
    """通常LOSO-CV。"""
    oof = np.zeros(len(y_orig)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=y_pow[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=y_pow[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(X[va_idx]), 0, None) ** (1 / P_POWER)
        iters.append(m.best_iteration)
    return loso_rmse(oof, y_orig), int(np.mean(iters))


# ── Load ───────────────────────────────────────────────────────────────────────
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

BASE = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "min_child_samples": 10}

# ── Phase1: 複数種holdout CV でパラメータ探索 ──────────────────────────────────
print(f"=== {EXP} Phase1: Multi-holdout CV (2〜3種同時) ===")
print(f"Fold構成: {MULTI_FOLDS}\n")

GRID = [
    {"num_leaves": 31,  "feature_fraction": 0.07},
    {"num_leaves": 63,  "feature_fraction": 0.07},  # P1
    {"num_leaves": 63,  "feature_fraction": 0.05},
    {"num_leaves": 63,  "feature_fraction": 0.10},
    {"num_leaves": 127, "feature_fraction": 0.07},
    {"num_leaves": 31,  "feature_fraction": 0.05},
]

print(f"{'leaves':>7} {'ff':>6}  {'MH-RMSE':>9}  {'avg_iter':>9}")
print("-" * 40)

best_mh_rmse = np.inf; best_g = None; best_avg_r = None

for g in GRID:
    params = {**BASE, **g}
    mh_rmse, avg_r = multi_holdout_cv(Xtr_epo, y_pow, y_train, sp_train, params, MULTI_FOLDS)
    tag = " <-- best" if mh_rmse < best_mh_rmse else ""
    p1_tag = " [P1]" if g["num_leaves"] == 63 and g["feature_fraction"] == 0.07 else ""
    print(f"  lv={g['num_leaves']:3d}  ff={g['feature_fraction']:.2f}  "
          f"{mh_rmse:9.4f}  {avg_r:9d}{p1_tag}{tag}")
    if mh_rmse < best_mh_rmse:
        best_mh_rmse = mh_rmse; best_g = g; best_avg_r = avg_r

print(f"\nMH best: leaves={best_g['num_leaves']}, ff={best_g['feature_fraction']:.2f}, "
      f"MH-RMSE={best_mh_rmse:.4f}")

# ── Phase2: best params を通常LOSO-CVで評価 ────────────────────────────────────
print(f"\n=== Phase2: Best params を LOSO-CV で再評価 ===")
best_params = {**BASE, **best_g}
p1_params   = {**BASE, "num_leaves": 63, "feature_fraction": 0.07}

loso_best, avg_r_best = loso_cv(Xtr_epo, y_pow, y_train, sp_train, best_params)
loso_p1,   avg_r_p1   = loso_cv(Xtr_epo, y_pow, y_train, sp_train, p1_params)

print(f"MH-best params  LOSO: {loso_best:.4f}  avg_iter={avg_r_best}")
print(f"P1 params       LOSO: {loso_p1:.4f}  avg_iter={avg_r_p1}  (参考: 実績15.4725)")
print(f"Delta                : {loso_best - loso_p1:+.4f}")

# ── 提出 ──────────────────────────────────────────────────────────────────────
final_params = best_params if loso_best < loso_p1 else p1_params
final_r = avg_r_best if loso_best < loso_p1 else avg_r_p1
final_loso = min(loso_best, loso_p1)

dtrain_f = lgb.Dataset(Xtr_epo, label=y_pow)
final = lgb.train(final_params, dtrain_f, num_boost_round=final_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: multi_holdout_best, LOSO={final_loso:.4f}", loso=final_loso)
print(f"[Done] {EXP}")
