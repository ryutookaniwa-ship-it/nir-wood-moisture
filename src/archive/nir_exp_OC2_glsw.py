"""
Experiment OC2: GLSW (Generalized Least Squares Weighting)
===========================================================
論文: Martens et al. 2003; 比較研究 2024 (Spectroscopy Letters)

EPO が「干渉方向を完全に射影除去（ハード）」するのに対し、
GLSW は正則化パラメータ alpha で「ソフトな重み付け減衰」を行う。

GLSW アルゴリズム:
1. 干渉変動の共分散行列 C を構築 (EPO と同じ入力)
2. フィルタ行列 W = (I + alpha * C)^(-1) を計算
3. X_filtered = X @ W  (soft filtering、完全除去ではない)

alpha が大きいほど干渉除去が強くなる (EPO の n_components に相当)。

Grid: alpha = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)

EXP     = "OC2"
P_POWER = 0.27
P1_LOSO = 15.4725

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)


def compute_glsw_filter(X, y, sp, alpha, bin_width=10.0):
    """
    GLSW フィルタ行列 W = (I + alpha * C)^(-1) を計算。
    C は EPO と同様に bin 内樹種間変動から構築する共分散行列。
    """
    n_feat = X.shape[1]
    C = np.zeros((n_feat, n_feat))

    bins = np.arange(0, y.max() + bin_width, bin_width)
    n_blocks = 0
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < 2: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        C += inter.T @ inter
        n_blocks += 1

    if n_blocks == 0:
        return np.eye(n_feat)

    C /= n_blocks
    # W = (I + alpha * C)^(-1)
    W = np.linalg.inv(np.eye(n_feat) + alpha * C)
    return W


def run_loso(Xtr, y, sp, params, tag):
    oof = np.zeros(len(y))
    best_iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y[tr_idx] ** P_POWER)
        dval   = lgb.Dataset(Xtr[va_idx], label=y[va_idx] ** P_POWER,
                             reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** (1 / P_POWER)
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y)
    avg_i = int(np.mean(best_iters))
    print(f"  {tag}: LOSO={rmse:.4f}  avg_iter={avg_i}  delta={rmse-P1_LOSO:+.4f}")
    return rmse, avg_i, oof


# ── データ ────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()

y_tr     = train[target_col].values
X_raw    = train[spec_cols].values.astype(np.float64)
sp_tr    = train["species number"].values
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values

# MSC + SG
ref    = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw,    ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)

print(f"P1ベースライン (EPO n=5): LOSO={P1_LOSO}")
print("GLSW grid探索中...\n")

# ── Grid探索 ──────────────────────────────────────────────────────────────────
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
results = []

for alpha in ALPHAS:
    print(f"=== alpha={alpha} ===")
    W = compute_glsw_filter(Xtr_sg, y_tr, sp_tr, alpha=alpha)
    Xtr_w = Xtr_sg @ W
    Xte_w = Xte_sg @ W
    rmse, avg_i, oof = run_loso(Xtr_w, y_tr, sp_tr, P1_PARAMS, f"GLSW alpha={alpha}")
    results.append((rmse, avg_i, alpha, Xtr_w, Xte_w))

# ── サマリ ────────────────────────────────────────────────────────────────────
print(f"\n=== OC2 グリッド結果サマリ ===")
results.sort(key=lambda x: x[0])
print(f"{'alpha':<12} {'LOSO':>8} {'delta':>8}")
print("-" * 32)
for rmse, avg_i, alpha, _, _ in results:
    mark = " <- BEST" if rmse == results[0][0] else ""
    print(f"{alpha:<12} {rmse:>8.4f} {rmse-P1_LOSO:>+8.4f}{mark}")

best_rmse, best_iter, best_alpha, best_Xtr, best_Xte = results[0]
print(f"\nBEST: alpha={best_alpha}")
print(f"  LOSO={best_rmse:.4f}  delta vs P1={best_rmse-P1_LOSO:+.4f}")

# ── 提出 ──────────────────────────────────────────────────────────────────────
dtrain_f = lgb.Dataset(best_Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=best_iter,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(best_Xte), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: GLSW alpha={best_alpha}, LOSO={best_rmse:.4f}",
                  loso=best_rmse)

print(f"\n[Done] {EXP}")
