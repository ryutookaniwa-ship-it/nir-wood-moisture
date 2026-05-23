"""
Experiment OB2: GroupSeq + EPO 相補性検証
==========================================
EPO (樹種間変動の除去) と GroupSeq (乾燥時系列の動態) は
解決する問題が直交しており、組み合わせが有効な可能性がある。

GS1/GS2 が P1 で悪化した原因の仮説:
  - position_ratio が樹種ごとに異なる MC 範囲を意味し混乱
  - delta_prev を生スペクトルから計算すると樹種固有ベースラインが乗る

本実験での改善策:
  OB2-A: EPO後スペクトルの delta_prev (樹種固有パターン除去済み)
  OB2-B: position_ratio/index を除外した delta_prev+rolling のみ (8列)
  OB2-C: OB2-A + OB2-B の組み合わせ (EPO後 delta_prev のみ3列)
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

EXP     = "OB2"
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
        sp_means = np.array([X[mask][sp[mask]==s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs:
        return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T


def apply_epo(X, V):
    return X - (X @ V) @ V.T


def delta_prev_features(df, X_spec, wns, col_5200=None, col_7000=None):
    """
    樹種内で sample_number 順にソートし、前回との差分を計算。
    X_spec: 各サンプルのスペクトル (同じ row 順で df と対応)
    返り値: (n_samples, 3) — delta_5200, delta_7000, delta_global_mean
    """
    if col_5200 is None:
        col_5200 = int(np.argmin(np.abs(wns - 5200)))
    if col_7000 is None:
        col_7000 = int(np.argmin(np.abs(wns - 7000)))

    n = len(df)
    feats = np.zeros((n, 3))
    df_reset = df.reset_index(drop=True)

    for sp, grp in df_reset.groupby("species number"):
        grp_sorted = grp.sort_values("sample number")
        idx = grp_sorted.index.tolist()
        spec = X_spec[idx]

        v5200 = spec[:, col_5200]
        v7000 = spec[:, col_7000]
        vglob = spec.mean(axis=1)

        feats[idx, 0] = np.concatenate([[0], np.diff(v5200)])
        feats[idx, 1] = np.concatenate([[0], np.diff(v7000)])
        feats[idx, 2] = np.concatenate([[0], np.diff(vglob)])

    return feats


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

# ── P1パイプライン (MSC+SG+EPO) ───────────────────────────────────────────────
ref    = X_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_te,  ref), window=9, polyorder=2)
V      = compute_epo(Xtr_sg, y_tr, sp_tr, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)
print(f"P1パイプライン完了: EPO後 {Xtr_epo.shape[1]}次元")

idx_5200 = int(np.argmin(np.abs(wns - 5200)))
idx_7000 = int(np.argmin(np.abs(wns - 7000)))
print(f"水分帯インデックス: 5200->col{idx_5200}, 7000->col{idx_7000}")

# ── OB2-A: EPO後スペクトルの delta_prev ──────────────────────────────────────
print("\n=== OB2-A: EPO後delta_prev(3列)+P1(1555列) ===")
gs_tr_A = delta_prev_features(train, Xtr_epo, wns, idx_5200, idx_7000)
gs_te_A = delta_prev_features(test,  Xte_epo, wns, idx_5200, idx_7000)
Xtr_A = np.hstack([Xtr_epo, gs_tr_A])
Xte_A = np.hstack([Xte_epo, gs_te_A])
print(f"  特徴量: {Xtr_A.shape[1]}次元")

oof_A = np.zeros(len(y_tr)); iters_A = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    dtrain = lgb.Dataset(Xtr_A[tr_idx], label=y_tr[tr_idx]**P_POWER)
    dval   = lgb.Dataset(Xtr_A[va_idx], label=y_tr[va_idx]**P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_A[va_idx] = np.clip(m.predict(Xtr_A[va_idx]), 0, None)**(1/P_POWER)
    iters_A.append(m.best_iteration)
rmse_A = loso_rmse(oof_A, y_tr)
avg_A  = int(np.mean(iters_A))
print(f"  OB2-A LOSO={rmse_A:.4f}  avg_iter={avg_A}  delta_vs_P1={rmse_A-15.4725:+.4f}")

# ── OB2-B: 生スペクトルの delta_prev (GS2 と同条件だが EPO と組み合わせ) ──────
print("\n=== OB2-B: 生スペクトルdelta_prev(3列)+P1(1555列) ===")
gs_tr_B = delta_prev_features(train, X_raw, wns, idx_5200, idx_7000)
gs_te_B = delta_prev_features(test,  X_te,  wns, idx_5200, idx_7000)
Xtr_B = np.hstack([Xtr_epo, gs_tr_B])
Xte_B = np.hstack([Xte_epo, gs_te_B])

oof_B = np.zeros(len(y_tr)); iters_B = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    dtrain = lgb.Dataset(Xtr_B[tr_idx], label=y_tr[tr_idx]**P_POWER)
    dval   = lgb.Dataset(Xtr_B[va_idx], label=y_tr[va_idx]**P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_B[va_idx] = np.clip(m.predict(Xtr_B[va_idx]), 0, None)**(1/P_POWER)
    iters_B.append(m.best_iteration)
rmse_B = loso_rmse(oof_B, y_tr)
avg_B  = int(np.mean(iters_B))
print(f"  OB2-B LOSO={rmse_B:.4f}  avg_iter={avg_B}  delta_vs_P1={rmse_B-15.4725:+.4f}")

# ── サマリ ────────────────────────────────────────────────────────────────────
print(f"\n=== {EXP} サマリ ===")
print(f"  P1   (EPOのみ):                LOSO=15.4725")
print(f"  GS2  (生delta_prev+P1, 旧実験): LOSO=15.8019")
print(f"  OB2-A(EPO後delta_prev+P1):     LOSO={rmse_A:.4f}  delta={rmse_A-15.4725:+.4f}")
print(f"  OB2-B(生delta_prev+P1):        LOSO={rmse_B:.4f}  delta={rmse_B-15.4725:+.4f}")

best_tag  = "OB2-A" if rmse_A <= rmse_B else "OB2-B"
best_rmse = min(rmse_A, rmse_B)
best_iter = avg_A if rmse_A <= rmse_B else avg_B
Xtr_best  = Xtr_A if rmse_A <= rmse_B else Xtr_B
Xte_best  = Xte_A if rmse_A <= rmse_B else Xte_B

dtrain_f = lgb.Dataset(Xtr_best, label=y_tr**P_POWER)
m_full = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=best_iter,
                   callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(m_full.predict(Xte_best), 0, None)**(1/P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT,
                  memo=f"{EXP}: {best_tag}, LOSO={best_rmse:.4f}",
                  loso=best_rmse)

print(f"\n[Done] {EXP}")
