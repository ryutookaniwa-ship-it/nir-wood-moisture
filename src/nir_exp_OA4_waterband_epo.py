"""
Experiment OA4: WaterBandSummary (EPO処理後スペクトルから抽出)
=============================================================
公式 WaterBandSummary: 7000±200 cm⁻¹ と 5200±200 cm⁻¹ 帯域の
{mean, std, min, max, area(台形積分)} 各5統計量 = 計10特徴量。

D4 との違い:
  D4: B2パイプライン + 生スペクトルから水バンド面積/比を抽出 → 改善なし
  OA4: P1パイプライン + EPO処理後スペクトルから公式WaterBandSummaryを抽出

EPO後のスペクトルは樹種間の散乱変動が除去されているため、
生スペクトルより水分含有量シグナルが純化されている可能性がある。
ただし D4 と同様に改善なしの見込み (EPOが既に処理済み)。
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

EXP     = "OA4"
P_POWER = 0.27

# 水吸収帯定義 (cm⁻¹)
WATER_BANDS = [(6800, 7200), (5000, 5400)]

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


def waterband_summary(X, wns, bands):
    """
    各帯域ごとに {mean, std, min, max, area} を抽出。
    Returns: (n_samples, len(bands)*5) array
    """
    feats = []
    for lo, hi in bands:
        mask = (wns >= lo) & (wns <= hi)
        seg = X[:, mask]
        wns_seg = wns[mask]
        means = seg.mean(axis=1)
        stds  = seg.std(axis=1)
        mins  = seg.min(axis=1)
        maxs  = seg.max(axis=1)
        # 台形積分 (波数軸方向)
        areas = np.trapezoid(seg, x=wns_seg, axis=1)
        feats.extend([means, stds, mins, maxs, areas])
    return np.column_stack(feats)


# ── データ ────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()
wns = np.array([float(c) for c in spec_cols])

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

# ── WaterBandSummary (EPO後スペクトルから) ────────────────────────────────────
# 波数軸を sg_deriv 後でも同じ列数なので wns をそのまま使う
Xtr_wb = waterband_summary(Xtr_epo, wns, WATER_BANDS)
Xte_wb = waterband_summary(Xte_epo, wns, WATER_BANDS)
Xtr = np.hstack([Xtr_epo, Xtr_wb])
Xte = np.hstack([Xte_epo, Xte_wb])
print(f"特徴量: EPO(1555) + WaterBandSummary({Xtr_wb.shape[1]}) = {Xtr.shape[1]}次元")
print(f"帯域: {WATER_BANDS}")

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
print(f"\nOA4 LOSO={rmse:.4f}  avg_iter={avg_i}")
print(f"P1  LOSO=15.4725  delta={rmse-15.4725:+.4f}")
print(f"D4 (生スペクトル水バンド) LOSO=16.44 改善なし (参考)")

# ── 全データ再学習 → テスト予測 ───────────────────────────────────────────────
dtrain_full = lgb.Dataset(Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_full, num_boost_round=avg_i,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(Xte), 0, None) ** (1/P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: P1+WaterBandSummary(EPO後,10特徴量), LOSO={rmse:.4f}",
                  loso=rmse)

print(f"\n[Done] {EXP}")
