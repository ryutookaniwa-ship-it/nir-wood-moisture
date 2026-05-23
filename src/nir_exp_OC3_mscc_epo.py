"""
Experiment OC3: MSCC基準によるEPO n_components最適化
====================================================
論文: "A simple but effective evaluation criterion for parameters optimization
      of EPO" Analytica Chimica Acta 2023

MSCC (Mean Spectral Correlation Coefficient):
  同じMC帯域内の異なる樹種サンプル間のスペクトル相関係数の平均。
  EPOが樹種変動を正しく除去できているほど、同MC帯の樹種間スペクトルが
  似るはずなので MSSCが高くなる。

現在はLOSO-CVでn=5を決定済みだが、MSSCという別の客観基準で確認する。
また、標準EPOとは異なるn値 (1〜10) でMSSC vs LOSO のトレードオフを可視化。

目的: n=5 が真に最適か、MSCC視点から検証する。
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

EXP     = "OC3"
P_POWER = 0.27
P1_LOSO = 15.4725

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)


def compute_epo(X, y, sp, n_components=5, bin_width=10.0):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < 2: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42)
        pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs:
        return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T


def apply_epo(X, V):
    return X - (X @ V) @ V.T


def compute_mscc(X_epo, y, sp, bin_width=10.0):
    """
    MSCC: 同一MC帯域内で異なる樹種のサンプル間スペクトル相関の平均。
    EPO後スペクトルで高いほど樹種変動が除去されている。
    """
    bins = np.arange(0, y.max() + bin_width, bin_width)
    corrs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < 2: continue
        # 異なる樹種ペアのサンプル間相関
        for i, s1 in enumerate(sp_in):
            for s2 in sp_in[i+1:]:
                idx1 = np.where(mask & (sp == s1))[0]
                idx2 = np.where(mask & (sp == s2))[0]
                # 各組み合わせの平均スペクトル同士の相関
                m1 = X_epo[idx1].mean(axis=0)
                m2 = X_epo[idx2].mean(axis=0)
                if m1.std() > 0 and m2.std() > 0:
                    r = np.corrcoef(m1, m2)[0, 1]
                    corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


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
    return rmse, avg_i


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

ref    = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw,    ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)

print(f"P1ベースライン (EPO n=5): LOSO={P1_LOSO}")
print(f"\nn_components vs MSCC vs LOSO:\n")
print(f"{'n':>4} {'MSCC':>8} {'LOSO':>8} {'delta':>8}")
print("-" * 35)

results = []
for n in range(1, 11):
    V = compute_epo(Xtr_sg, y_tr, sp_tr, n_components=n)
    Xtr_epo = apply_epo(Xtr_sg, V)
    Xte_epo = apply_epo(Xte_sg, V)
    mscc = compute_mscc(Xtr_epo, y_tr, sp_tr)
    rmse, avg_i = run_loso(Xtr_epo, y_tr, sp_tr, P1_PARAMS, f"n={n}")
    mark = " *" if n == 5 else ""
    print(f"{n:>4} {mscc:>8.4f} {rmse:>8.4f} {rmse-P1_LOSO:>+8.4f}{mark}")
    results.append((n, mscc, rmse, avg_i, Xtr_epo, Xte_epo))

# MSSCとLOSOで最良の n を比較
best_by_mscc = max(results, key=lambda x: x[1])
best_by_loso = min(results, key=lambda x: x[2])

print(f"\nMSSC最大: n={best_by_mscc[0]}  MSCC={best_by_mscc[1]:.4f}  LOSO={best_by_mscc[2]:.4f}")
print(f"LOSO最小: n={best_by_loso[0]}  MSCC={best_by_loso[1]:.4f}  LOSO={best_by_loso[2]:.4f}")
print(f"P1(n=5): MSCC={results[4][1]:.4f}  LOSO={results[4][2]:.4f}")

best_n, best_mscc, best_rmse, best_iter, best_Xtr, best_Xte = best_by_loso

dtrain_f = lgb.Dataset(best_Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=best_iter,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(best_Xte), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: MSCC最適n={best_n}, LOSO={best_rmse:.4f}",
                  loso=best_rmse)

print(f"\n[Done] {EXP}")
