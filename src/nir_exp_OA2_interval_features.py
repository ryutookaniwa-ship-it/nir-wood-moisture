"""
Experiment OA2: P1 + IntervalMean/Slope (公式 interval_features.py 準拠)
=========================================================================
公式リポジトリの IntervalMean/Slope: 1555波長を15区間に分割し、
各区間の平均値と線形傾きを特徴量化 (計30特徴量)。

これをEPO処理後の1555次元特徴量に追加 → 1585次元でLGBM学習。
EPO後の特徴量から粗いスペクトル形状情報を追加する。

D4(水吸収帯エリア/比)が改善なしだったため期待値は低いが、
IntervalMean/Slope は全波長にわたる粗いスケール特徴量なので
性質が異なる可能性がある。
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

EXP     = "OA2"
P_POWER = 0.27
N_INTERVALS = 15

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


def interval_features(X, n_intervals=15):
    """
    1555次元を n_intervals 区間に分割し、各区間の平均値と線形傾きを返す。
    Returns: (n_samples, n_intervals*2) array
    """
    n_feat = X.shape[1]
    edges = np.linspace(0, n_feat, n_intervals + 1, dtype=int)
    means  = []
    slopes = []
    for i in range(n_intervals):
        seg = X[:, edges[i]:edges[i+1]]
        means.append(seg.mean(axis=1))
        # 線形傾き: polyfit の係数[0]
        x = np.arange(seg.shape[1], dtype=float)
        if seg.shape[1] > 1:
            # 各サンプルの傾き: (x - xmean)·seg / sum((x-xmean)^2)
            xm = x - x.mean()
            denom = (xm ** 2).sum()
            sl = (seg @ xm) / denom if denom > 0 else np.zeros(len(X))
        else:
            sl = np.zeros(len(X))
        slopes.append(sl)
    return np.column_stack(means + slopes)


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

# ── P1パイプライン ─────────────────────────────────────────────────────────────
ref    = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te,  ref), window=9, polyorder=2)
V      = compute_epo(Xtr_sg, y_tr, sp_tr, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── IntervalMean/Slope 追加 ────────────────────────────────────────────────────
Xtr_iv = interval_features(Xtr_epo, N_INTERVALS)
Xte_iv = interval_features(Xte_epo, N_INTERVALS)
Xtr = np.hstack([Xtr_epo, Xtr_iv])
Xte = np.hstack([Xte_epo, Xte_iv])
print(f"特徴量: EPO(1555) + IntervalMean/Slope({N_INTERVALS*2}) = {Xtr.shape[1]}次元")

# ── LOSO-CV ───────────────────────────────────────────────────────────────────
oof = np.zeros(len(y_tr))
best_iters = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_tr[tr_idx] ** P_POWER)
    dval   = lgb.Dataset(Xtr[va_idx], label=y_tr[va_idx] ** P_POWER,
                         reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** (1/P_POWER)
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_tr)
avg_i = int(np.mean(best_iters))
print(f"\nOA2 LOSO={rmse:.4f}  avg_iter={avg_i}")
print(f"P1  LOSO=15.4725  delta={rmse-15.4725:+.4f}")

# ── 全データ再学習 → テスト予測 ───────────────────────────────────────────────
dtrain_full = lgb.Dataset(Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_full, num_boost_round=avg_i,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(Xte), 0, None) ** (1/P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: P1+IntervalMean/Slope({N_INTERVALS*2}特徴量), LOSO={rmse:.4f}",
                  loso=rmse)

print(f"\n[Done] {EXP}")
