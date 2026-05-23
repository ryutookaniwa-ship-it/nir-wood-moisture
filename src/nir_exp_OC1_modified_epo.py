"""
Experiment OC1: Modified EPO (y相関フィルタリング)
===================================================
論文: "The modified external parameter orthogonalization with removed PC2"
      Geoderma 2024, doi:10.1016/j.geoderma.2024.116802

標準EPOは樹種間変動の上位n方向をすべて除去するが、
その中に含水率と相関する方向が混入すると有益シグナルを捨てる。

Modified EPO:
1. 多めにEPO方向を計算 (n_candidate)
2. 各方向のy相関を測定
3. |corr| < threshold の方向のみを除去対象とする
   (|corr| >= threshold の方向は保持 = 含水率シグナルとみなす)

Grid:
  n_candidate: [10, 15, 20]
  threshold:   [0.1, 0.2, 0.3, 0.4]
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

EXP     = "OC1"
P_POWER = 0.27
P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

# P1ベースライン (標準EPO n=5)
P1_LOSO = 15.4725


def compute_inter_species_directions(X, y, sp, bin_width=10.0, n_candidate=20):
    """樹種間変動の候補方向を多めに計算する (標準EPOより多く)"""
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
        n_c = min(n_candidate, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42)
        pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs:
        return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_candidate].T  # (n_features, n_candidate)


def compute_modified_epo(X, y, sp, bin_width=10.0, n_candidate=20, threshold=0.3):
    """
    Modified EPO: y相関が低い方向のみ除去。
    Returns V_remove: 除去する方向の行列 (n_features, n_remove)
    """
    V_all = compute_inter_species_directions(X, y, sp, bin_width, n_candidate)
    n_dirs = V_all.shape[1]

    # 各方向のy相関
    scores = X @ V_all  # (n_samples, n_candidate)
    corrs = np.array([np.corrcoef(scores[:, i], y)[0, 1] for i in range(n_dirs)])

    # |corr| < threshold の方向のみ除去対象
    remove_mask = np.abs(corrs) < threshold
    n_remove = remove_mask.sum()

    print(f"    候補{n_dirs}方向 → 除去{n_remove}方向 "
          f"(保持{n_dirs-n_remove}方向, |corr|>={threshold:.1f})")
    print(f"    各方向の|corr|: {np.abs(corrs[:min(n_dirs,10)]).round(3).tolist()}")

    if n_remove == 0:
        return np.zeros((X.shape[1], 1))
    return V_all[:, remove_mask]


def apply_epo(X, V):
    return X - (X @ V) @ V.T


def run_loso(Xtr, y, sp, Xte, params, tag):
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
    print(f"    {tag}: LOSO={rmse:.4f}  avg_iter={avg_i}  delta={rmse-P1_LOSO:+.4f}")
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

# MSC + SG (P1前処理)
ref    = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw,    ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)

# ── Grid探索 ─────────────────────────────────────────────────────────────────
print(f"P1ベースライン (標準EPO n=5): LOSO={P1_LOSO}")
print()

results = []
for n_cand in [10, 15, 20]:
    for thresh in [0.1, 0.2, 0.3, 0.4]:
        tag = f"n={n_cand},thr={thresh}"
        print(f"=== OC1 {tag} ===")
        V = compute_modified_epo(Xtr_sg, y_tr, sp_tr,
                                 n_candidate=n_cand, threshold=thresh)
        Xtr_epo = apply_epo(Xtr_sg, V)
        Xte_epo = apply_epo(Xte_sg, V)
        rmse, avg_i, oof = run_loso(Xtr_epo, y_tr, sp_tr, Xte_epo,
                                    P1_PARAMS, tag)
        results.append((rmse, avg_i, n_cand, thresh, Xtr_epo, Xte_epo))
        print()

# 標準EPO (n=5) の確認
print("=== 標準EPO n=5 (P1) の確認 ===")
from sklearn.decomposition import PCA as _PCA

def std_epo(X, y, sp, n=5):
    bins = np.arange(0, y.max() + 10.0, 10.0)
    all_dirs = []
    for lo in bins[:-1]:
        mask = (y >= lo) & (y < lo + 10.0)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < 2: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = _PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n].T

V_std = std_epo(Xtr_sg, y_tr, sp_tr, n=5)
scores_std = Xtr_sg @ V_std
corrs_std = [np.corrcoef(scores_std[:, i], y_tr)[0, 1] for i in range(5)]
print(f"  標準EPO n=5方向の|corr|: {np.abs(corrs_std).round(3).tolist()}")

# ── サマリ ────────────────────────────────────────────────────────────────────
print("\n=== OC1 グリッド結果サマリ ===")
results.sort(key=lambda x: x[0])
print(f"{'設定':<25} {'LOSO':>8} {'delta':>8}")
print("-" * 45)
for rmse, avg_i, n_c, thr, _, _ in results:
    tag = f"n={n_c},thr={thr}"
    mark = " ← BEST" if rmse == results[0][0] else ""
    print(f"{tag:<25} {rmse:>8.4f} {rmse-P1_LOSO:>+8.4f}{mark}")

best_rmse, best_iter, best_nc, best_thr, best_Xtr, best_Xte = results[0]
print(f"\nBEST: n_candidate={best_nc}, threshold={best_thr}")
print(f"  LOSO={best_rmse:.4f}  delta vs P1={best_rmse-P1_LOSO:+.4f}")

# ── 提出 ──────────────────────────────────────────────────────────────────────
dtrain_f = lgb.Dataset(best_Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=best_iter,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(best_Xte), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: ModifiedEPO n={best_nc} thr={best_thr}, LOSO={best_rmse:.4f}",
                  loso=best_rmse)

print(f"\n[Done] {EXP}")
