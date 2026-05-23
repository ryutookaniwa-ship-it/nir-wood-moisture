"""
Experiment OA1: log1p ターゲット変換 (公式レシピ snv_lgbm_log1p 準拠)
======================================================================
公式リポジトリは log1p 変換を使用。我々は p=0.27 に最適化済みだが、
log1p との直接比較が未実施のため実験。

比較:
  p=0.27: MC=100% → 3.631 (高MC域を強く圧縮)
  log1p : MC=100% → 4.615 (中間域をより均等に扱う)

P1パイプライン (MSC+SG(w=9,p=2)+EPO(n=5)) は同一。変換のみ変更。
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
    TRAIN_PATH, TEST_PATH, BASE_DIR, LOSO_SUBMIT_THRESHOLD,
)

EXP = "OA1"

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)


def compute_epo(X, y, sp, bin_width=10.0, n_components=5):
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


# ── データ ────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()

y_tr   = train[target_col].values
X_raw  = train[spec_cols].values.astype(np.float64)
sp_tr  = train["species number"].values
X_te   = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values

# ── P1パイプライン (MSC+SG+EPO) ───────────────────────────────────────────────
ref   = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te,  ref), window=9, polyorder=2)
V      = compute_epo(Xtr_sg, y_tr, sp_tr, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)
print(f"Pipeline: MSC+SG(w=9,p=2)+EPO(n=5) → {Xtr.shape[1]}次元")

# ── LOSO: log1p変換 ───────────────────────────────────────────────────────────
print("\n=== OA1-A: log1p変換 ===")
oof_log1p = np.zeros(len(y_tr))
best_iters_log1p = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=np.log1p(y_tr[tr_idx]))
    dval   = lgb.Dataset(Xtr[va_idx], label=np.log1p(y_tr[va_idx]),
                         reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof_log1p[va_idx] = np.clip(np.expm1(m.predict(Xtr[va_idx])), 0, None)
    best_iters_log1p.append(m.best_iteration)

rmse_log1p = loso_rmse(oof_log1p, y_tr)
avg_i_log1p = int(np.mean(best_iters_log1p))
print(f"  log1p  LOSO={rmse_log1p:.4f}  avg_iter={avg_i_log1p}")
print(f"  p=0.27 LOSO=15.4725  (P1ベースライン)")
print(f"  delta={rmse_log1p-15.4725:+.4f}")

# ── 全データ再学習 → テスト予測 ───────────────────────────────────────────────
dtrain_full = lgb.Dataset(Xtr, label=np.log1p(y_tr))
m_full = lgb.train(P1_PARAMS, dtrain_full, num_boost_round=avg_i_log1p,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(np.expm1(m_full.predict(Xte)), 0, None)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)

submit_to_signate(OUT,
                  memo=f"{EXP}: log1p変換, LOSO={rmse_log1p:.4f}",
                  loso=rmse_log1p)

print(f"\n[Done] {EXP}")
