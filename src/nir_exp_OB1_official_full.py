"""
Experiment OB1: 公式フルパイプライン LOSO評価
=============================================
公式リポジトリ (hirokenn/spectral_analysis) の完全なパイプラインを
我々の LOSO-CV で評価する。これまで個別特徴量を P1 に追加する形でしか
試していなかったが、公式特徴量群は互いを補完する設計。

公式パイプライン (snv_lgbm レシピ):
  SNV → [IntervalMean(15) + IntervalSlope(15) + DWT要約(12)
          + WaterBandSummary(10) + GroupSeq(11)] = 63特徴量
  → LGBM(colsample=0.8, leaves=31, lr=0.03)

バリアント:
  OB1-A: 13種全体 + log1p
  OB1-B: sp15/17除外(11種) + log1p  ← 公式設定に最も近い
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    snv, loso_folds, loso_rmse,
    save_submission, submit_to_signate,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)

EXP = "OB1"
EXCLUDE_SP = [15, 17]

OFFICIAL_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42,
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
)


# ── 特徴量エンジニアリング ────────────────────────────────────────────────────
def interval_features(X, n=15):
    edges = np.linspace(0, X.shape[1], n + 1, dtype=int)
    means, slopes = [], []
    for i in range(n):
        seg = X[:, edges[i]:edges[i+1]]
        means.append(seg.mean(axis=1))
        if seg.shape[1] > 1:
            x = np.arange(seg.shape[1], dtype=float)
            xm = x - x.mean()
            denom = (xm**2).sum()
            slopes.append((seg @ xm) / denom if denom > 0 else np.zeros(len(X)))
        else:
            slopes.append(np.zeros(len(X)))
    return np.column_stack(means + slopes)


def dwt_summary(X, wavelet="db4", level=3):
    if not HAS_PYWT:
        return np.zeros((len(X), 12))
    feats = []
    for row in X:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        row_feats = []
        for c in coeffs:
            row_feats.extend([c.mean(), c.std(), np.dot(c, c)])
        feats.append(row_feats)
    return np.array(feats)


def waterband_summary(X, wns):
    bands = [(6800, 7200), (5000, 5400)]
    feats = []
    for lo, hi in bands:
        mask = (wns >= lo) & (wns <= hi)
        seg = X[:, mask]
        wns_seg = wns[mask]
        feats.extend([seg.mean(axis=1), seg.std(axis=1),
                      seg.min(axis=1), seg.max(axis=1),
                      np.trapezoid(seg, x=wns_seg, axis=1)])
    return np.column_stack(feats)


def group_seq_features(df, spec_cols, wns):
    """公式 GroupSequenceFeatures の完全実装"""
    idx_5200 = int(np.argmin(np.abs(wns - 5200)))
    idx_7000 = int(np.argmin(np.abs(wns - 7000)))

    df = df.sort_values(["species number", "sample number"]).copy()
    feats = np.zeros((len(df), 11))

    for sp, grp in df.groupby("species number"):
        idx = grp.index
        n = len(idx)
        spec = grp[spec_cols].values

        feats[df.index.get_indexer(idx), 0] = np.arange(1, n + 1)          # position_index
        feats[df.index.get_indexer(idx), 1] = np.linspace(0, 1, n)         # position_ratio

        v5200 = spec[:, idx_5200]
        v7000 = spec[:, idx_7000]
        vglob = spec.mean(axis=1)

        delta5 = np.concatenate([[0], np.diff(v5200)])
        delta7 = np.concatenate([[0], np.diff(v7000)])
        deltag = np.concatenate([[0], np.diff(vglob)])
        feats[df.index.get_indexer(idx), 2] = delta5
        feats[df.index.get_indexer(idx), 3] = delta7
        feats[df.index.get_indexer(idx), 4] = deltag

        w = 5
        for j, (v, base) in enumerate([(v5200, 5), (v7000, 7), (vglob, 9)]):
            rm = np.array([v[max(0, i-w+1):i+1].mean() for i in range(n)])
            rs = np.array([v[max(0, i-w+1):i+1].std() if i >= 1 else 0.0
                           for i in range(n)])
            feats[df.index.get_indexer(idx), base]   = rm
            feats[df.index.get_indexer(idx), base+1] = rs

    return feats


# ── データ ────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()
wns = np.array([float(c) for c in spec_cols])

y_all  = train[target_col].values
X_all  = train[spec_cols].values.astype(np.float64)
sp_all = train["species number"].values
X_te   = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values

# ── 公式特徴量 (全13種) ───────────────────────────────────────────────────────
X_snv_tr = snv(X_all)
X_snv_te = snv(X_te)

iv_tr = interval_features(X_snv_tr)
iv_te = interval_features(X_snv_te)
dw_tr = dwt_summary(X_snv_tr)
dw_te = dwt_summary(X_snv_te)
wb_tr = waterband_summary(X_snv_tr, wns)
wb_te = waterband_summary(X_snv_te, wns)
gs_tr = group_seq_features(train, spec_cols, wns)
gs_te = group_seq_features(test,  spec_cols, wns)

X_full_tr = np.hstack([iv_tr, dw_tr, wb_tr, gs_tr])
X_full_te = np.hstack([iv_te, dw_te, wb_te, gs_te])
print(f"公式特徴量: {X_full_tr.shape[1]}次元  "
      f"(IntervalMean/Slope={iv_tr.shape[1]}, DWT={dw_tr.shape[1]}, "
      f"WaterBand={wb_tr.shape[1]}, GroupSeq={gs_tr.shape[1]})")


def run_loso(X, y, sp, params, tag):
    oof = np.zeros(len(y))
    best_iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=np.log1p(y[tr_idx]))
        dval   = lgb.Dataset(X[va_idx], label=np.log1p(y[va_idx]),
                             reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(np.expm1(m.predict(X[va_idx])), 0, None)
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y)
    avg_i = int(np.mean(best_iters))
    print(f"  {tag}: LOSO={rmse:.4f}  avg_iter={avg_i}")
    return rmse, avg_i, oof


# ── OB1-A: 全13種 ─────────────────────────────────────────────────────────────
print("\n=== OB1-A: 公式フル(13種, log1p) ===")
rmse_A, iter_A, oof_A = run_loso(
    X_full_tr, y_all, sp_all, OFFICIAL_PARAMS, "OB1-A")

# ── OB1-B: sp15/17除外 ────────────────────────────────────────────────────────
print("\n=== OB1-B: 公式フル(sp15/17除外, log1p) ===")
keep = ~np.isin(sp_all, EXCLUDE_SP)
rmse_B, iter_B, oof_B = run_loso(
    X_full_tr[keep], y_all[keep], sp_all[keep], OFFICIAL_PARAMS, "OB1-B")

# ── サマリ ────────────────────────────────────────────────────────────────────
print(f"\n=== {EXP} サマリ ===")
print(f"  P1  (EPOあり,13種,p=0.27): LOSO=15.4725  LB=15.395")
print(f"  G   (EPOなし,13種,raw):     LOSO=21.48    LB=18.995")
print(f"  OB1-A(公式全63特徴量,13種): LOSO={rmse_A:.4f}")
print(f"  OB1-B(公式全63特徴量,11種): LOSO={rmse_B:.4f}  (11種比較)")

# 最良を提出
best_tag  = "OB1-A" if rmse_A <= rmse_B else "OB1-B"
best_rmse = min(rmse_A, rmse_B)
best_iter = iter_A if rmse_A <= rmse_B else iter_B

if best_tag == "OB1-A":
    dtrain_f = lgb.Dataset(X_full_tr, label=np.log1p(y_all))
    X_submit = X_full_te
else:
    keep = ~np.isin(sp_all, EXCLUDE_SP)
    dtrain_f = lgb.Dataset(X_full_tr[keep], label=np.log1p(y_all[keep]))
    X_submit = X_full_te

m_full = lgb.train(OFFICIAL_PARAMS, dtrain_f, num_boost_round=best_iter,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(np.expm1(m_full.predict(X_submit)), 0, None)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
from nir_loso_utils import save_submission
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: 公式全63特徴量({best_tag}), LOSO={best_rmse:.4f}",
                  loso=best_rmse)

print(f"\n[Done] {EXP}")
