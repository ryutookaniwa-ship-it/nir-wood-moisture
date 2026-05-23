"""
Experiment EX2: EPOなし + sp15(ベイスギ)/sp17(ベイマツ)除外
=============================================================
仮説: EPOなし環境では sp15/17 が損失を引きずる。除外すると残り11種に
      集中でき、テスト種との距離が縮まる可能性がある。
      また PL2 との誤差相関が低ければアンサンブル素材として有用。

比較ベースライン:
  G   (EPOなし, 13種): LB=18.995
  P1  (EPOあり, 13種): LOSO=15.4725, LB=15.395
  PL2 (EPOあり+疑似): LOSO=15.0623, LB=15.392
  EX1 (EPOあり, 11種): LB=18.819
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

EXP       = "EX2"
P_POWER   = 0.27
EXCLUDE_SP = [15, 17]

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

# ── データ ────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()

y_all  = train[target_col].values
X_all  = train[spec_cols].values.astype(np.float64)
sp_all = train["species number"].values
X_te   = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_te  = test["species number"].values

# ── 除外 ─────────────────────────────────────────────────────────────────────
keep = ~np.isin(sp_all, EXCLUDE_SP)
X_tr_raw = X_all[keep]
y_tr     = y_all[keep]
sp_tr    = sp_all[keep]
print(f"全訓練: {len(y_all)} → 除外後: {len(y_tr)} ({len(np.unique(sp_tr))}種)")
for s in EXCLUDE_SP:
    print(f"  除外: sp{s} ({(sp_all==s).sum()}サンプル)")

# ── MSC + SG (EPOなし) ───────────────────────────────────────────────────────
ref       = X_tr_raw.mean(axis=0)
X_tr_pre  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
X_te_pre  = sg_deriv(msc(X_te,    ref), window=9, polyorder=2)
print(f"\n前処理: MSC(ref=訓練平均)+SG(w=9,p=2) → {X_tr_pre.shape[1]}次元 ※EPOなし")

# ── LOSO-CV ──────────────────────────────────────────────────────────────────
oof = np.zeros(len(y_tr))
best_iters = []
for tr_idx, va_idx, sp_out in loso_folds(sp_tr):
    dtrain = lgb.Dataset(X_tr_pre[tr_idx], label=y_tr[tr_idx] ** P_POWER)
    dval   = lgb.Dataset(X_tr_pre[va_idx], label=y_tr[va_idx] ** P_POWER,
                         reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(X_tr_pre[va_idx]), 0, None) ** (1/P_POWER)
    best_iters.append(m.best_iteration)

rmse  = loso_rmse(oof, y_tr)
avg_i = int(np.mean(best_iters))
print(f"\nEX2 LOSO={rmse:.4f}  avg_iter={avg_i}")
print(f"  P1  (EPOあり,13種): 15.4725")
print(f"  G   (EPOなし,13種): 21.48")
print(f"  EX1 (EPOあり,11種): LB=18.819")

# 種別RMSE
for s in sorted(np.unique(sp_tr)):
    idx = np.where(sp_tr == s)[0]
    sp_r = np.sqrt(np.mean((y_tr[idx]-oof[idx])**2))
    print(f"  sp{s:2d}: RMSE={sp_r:.2f}")

# ── 全データ再学習 → テスト予測 ───────────────────────────────────────────────
dtrain_full = lgb.Dataset(X_tr_pre, label=y_tr ** P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_full, num_boost_round=avg_i,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(X_te_pre), 0, None) ** (1/P_POWER)

# ── PL2との相関確認 ──────────────────────────────────────────────────────────
pl2_path = str(BASE_DIR / "output" / "nir-wood-moisture" / "submission_PL2.csv")
try:
    pl2 = pd.read_csv(pl2_path, header=None, names=["id","pred"])
    r = np.corrcoef(preds, pl2["pred"].values)[0, 1]
    print(f"\nPL2テスト予測との相関 r={r:.4f}")
    print(f"  r<0.99 → アンサンブル多様性あり" if r < 0.99 else
          f"  r≥0.99 → P1/PL2と同質、ブレンド効果は限定的")
    # alpha探索
    print("\nブレンドPL2+EX2 (テスト予測相関のみ、参考値):")
    for alpha in [0.1, 0.2, 0.3]:
        blend = alpha * preds + (1-alpha) * pl2["pred"].values
        print(f"  alpha={alpha:.1f}: blend_mean={blend.mean():.2f}")
except Exception as e:
    print(f"\n[INFO] PL2読み込みスキップ: {e}")

# ── 保存 ─────────────────────────────────────────────────────────────────────
OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)

# EX2はアンサンブル評価目的のためLOSO閾値チェックをスキップして提出
print(f"\n[EX2] アンサンブル評価のため閾値チェックなしで提出")
submit_to_signate(OUT, memo=f"{EXP}: EPOなし+sp15/17除外, LOSO={rmse:.4f}")

print(f"\n[Done] {EXP}")
