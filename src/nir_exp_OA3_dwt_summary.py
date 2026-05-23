"""
Experiment OA3: P1 + DWT要約統計 (公式 dwt_features.py 準拠)
=============================================================
公式: db4, level=3 → 4サブバンド(a3, d3, d2, d1) × {mean, std, energy} = 12特徴量

V1a では db4 の全係数を追加 (大量の次元) → LOSO=15.5416 (+0.07)
本実験は12要約統計のみ追加 → 次元は1555+12=1567

V1aと違い「係数全体」ではなく「サブバンドの統計サマリ」のみ。
情報量は少ないが訓練樹種固有のスペクトル微細構造を記憶しにくいため、
LB-LOSOギャップが小さい可能性がある。
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[WARNING] pywt not installed. Install with: pip install PyWavelets")

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)

EXP     = "OA3"
P_POWER = 0.27

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


def dwt_summary(X, wavelet="db4", level=3):
    """
    各サンプルについて DWT(db4, level=3) を計算し
    4サブバンド × {mean, std, energy} = 12統計量を返す。
    """
    if not HAS_PYWT:
        return np.zeros((len(X), 12))
    feats = []
    for row in X:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        # coeffs[0]=a3, coeffs[1]=d3, coeffs[2]=d2, coeffs[3]=d1
        row_feats = []
        for c in coeffs:
            row_feats.extend([c.mean(), c.std(), np.dot(c, c)])
        feats.append(row_feats)
    return np.array(feats)


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

# ── DWT要約統計追加 ────────────────────────────────────────────────────────────
print("DWT要約統計計算中 (EPO処理後スペクトルに適用) ...")
Xtr_dwt = dwt_summary(Xtr_epo)
Xte_dwt = dwt_summary(Xte_epo)
Xtr = np.hstack([Xtr_epo, Xtr_dwt])
Xte = np.hstack([Xte_epo, Xte_dwt])
print(f"特徴量: EPO(1555) + DWT要約({Xtr_dwt.shape[1]}) = {Xtr.shape[1]}次元")
if not HAS_PYWT:
    print("[WARNING] PyWaveletsなし → DWT特徴量はゼロ(EPO単体と同等)")

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
print(f"\nOA3 LOSO={rmse:.4f}  avg_iter={avg_i}")
print(f"P1  LOSO=15.4725  delta={rmse-15.4725:+.4f}")
print(f"V1a (全db4係数) LOSO=15.5416  (参考)")

# ── 全データ再学習 → テスト予測 ───────────────────────────────────────────────
dtrain_full = lgb.Dataset(Xtr, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_full, num_boost_round=avg_i,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(Xte), 0, None) ** (1/P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: P1+DWT要約統計(12特徴量), LOSO={rmse:.4f}",
                  loso=rmse)

print(f"\n[Done] {EXP}")
