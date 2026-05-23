"""
Experiment GS1: GroupSequenceFeature + P1 pipeline
===================================================
物理的背景:
  各試料は飽水状態から室温乾燥しながら繰り返し測定。
  sample_id の樹種内順序 = 乾燥プロセスの時系列。
  → 乾燥進行位置・乾燥速度・近傍移動平均を特徴量化。

追加特徴量 (12列):
  group_position_index  : 樹種内での測定順位
  group_position_ratio  : 正規化順位 (0=最初, 1=最後)
  delta_prev_5200       : 前回測定との5200cm⁻¹差分
  delta_prev_7000       : 前回測定との7000cm⁻¹差分
  delta_prev_global_mean: 前回測定との全波長平均差分
  rolling_mean_5_{5200,7000,global}: 直近5測定の移動平均
  rolling_std_5_{5200,7000,global} : 直近5測定の移動標準偏差

ベース: P1 (MSC+SG(w=9,p=2)+EPO(n=5)+y^0.27+LGBM, LB=15.395, LOSO=15.4725)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
    TRAIN_PATH, TEST_PATH, BASE_DIR, LOSO_SUBMIT_THRESHOLD,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP = "GS1"
TARGET_WNS = (5200.0, 7000.0)
ROLLING_WINDOW = 5
P_POWER = 0.27

P1_PARAMS = {
    **LGBM_BASE_PARAMS,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
}


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4:
            continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species:
            continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1:
            continue
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


def compute_group_seq_features(X_sg, sample_ids, species, wns,
                                target_wns=TARGET_WNS,
                                rolling_window=ROLLING_WINDOW):
    """
    樹種内の sample_id 順序から時系列特徴量を生成。
    X_sg: MSC+SG 処理済スペクトル (EPO前)。物理的変化を保持。
    戻り値: (n_samples, 12) の特徴量行列
    """
    n_samples = X_sg.shape[0]
    target_indices = [int(np.argmin(np.abs(wns - t))) for t in target_wns]

    # 系列値: [5200cm⁻¹, 7000cm⁻¹, 全波長平均]
    series = np.column_stack(
        [X_sg[:, idx] for idx in target_indices] + [np.mean(X_sg, axis=1)]
    ).astype(np.float32)
    n_series = series.shape[1]

    pos_index = np.zeros(n_samples, dtype=np.float32)
    pos_ratio = np.zeros(n_samples, dtype=np.float32)
    delta_prev = np.zeros((n_samples, n_series), dtype=np.float32)
    roll_mean = np.zeros((n_samples, n_series), dtype=np.float32)
    roll_std = np.zeros((n_samples, n_series), dtype=np.float32)

    for sp in np.unique(species):
        grp_idx = np.where(species == sp)[0]
        order = np.argsort(sample_ids[grp_idx], kind="stable")
        ordered = grp_idx[order]
        g_size = len(ordered)

        for pos, idx in enumerate(ordered):
            pos_index[idx] = float(pos + 1)
            pos_ratio[idx] = 0.0 if g_size == 1 else float(pos / (g_size - 1))

            if pos > 0:
                delta_prev[idx] = series[idx] - series[ordered[pos - 1]]

            w_start = max(0, pos - rolling_window + 1)
            w_vals = series[ordered[w_start:pos + 1]]
            roll_mean[idx] = np.mean(w_vals, axis=0)
            roll_std[idx] = np.std(w_vals, axis=0)

    return np.column_stack([pos_index, pos_ratio, delta_prev, roll_mean, roll_std])


# ── Load ───────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")

target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()
wns        = np.array([float(c) for c in spec_cols])

y_train    = train[target_col].values
X_tr_raw   = train[spec_cols].values.astype(np.float64)
X_te_raw   = test[spec_cols].values.astype(np.float64)
train_ids  = train["sample number"].values
test_ids   = test["sample number"].values
sp_train   = train["species number"].values
sp_test    = test["species number"].values
y_pow      = y_train ** P_POWER

# ── P1 前処理 ─────────────────────────────────────────────────────────────────
ref      = X_tr_raw.mean(axis=0)
Xtr_sg   = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg   = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V        = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo  = apply_epo(Xtr_sg, V)
Xte_epo  = apply_epo(Xte_sg, V)

# ── GroupSequenceFeatures (MSC+SG後・EPO前のスペクトルを使用) ──────────────────
gs_tr = compute_group_seq_features(Xtr_sg, train_ids, sp_train, wns)
gs_te = compute_group_seq_features(Xte_sg, test_ids,  sp_test,  wns)

print(f"GroupSeq feature shape: {gs_tr.shape}")  # (1322, 12)
print(f"Feature names: position_index, position_ratio, "
      f"delta_5200, delta_7000, delta_mean, "
      f"roll_mean×3, roll_std×3")

# ── 結合 ─────────────────────────────────────────────────────────────────────
Xtr_full = np.hstack([Xtr_epo, gs_tr])  # (1322, 1567)
Xte_full = np.hstack([Xte_epo, gs_te])
print(f"Full feature shape: {Xtr_full.shape}")

# ── LOSO-CV ───────────────────────────────────────────────────────────────────
print(f"\n=== Experiment {EXP}: GroupSequenceFeature + P1 ===")
print(f"Baseline P1: LOSO=15.4725, LB=15.395\n")

oof = np.zeros(len(y_train))
best_iters = []

for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_full[tr_idx], label=y_pow[tr_idx])
    dval   = lgb.Dataset(Xtr_full[va_idx], label=y_pow[va_idx], reference=dtrain)
    m = lgb.train(
        P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof[va_idx] = np.clip(m.predict(Xtr_full[va_idx]), 0, None) ** (1 / P_POWER)
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train)
avg_r = int(np.mean(best_iters))

print(f"GS1 LOSO-RMSE : {rmse:.4f}")
print(f"P1  LOSO-RMSE : 15.4725")
print(f"Delta         : {rmse - 15.4725:+.4f}")
print(f"avg_iter      : {avg_r}")

print("\nPer-species RMSE:")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    sp_rmse = float(np.sqrt(np.mean((y_train[idx] - oof[idx]) ** 2)))
    marker = " ★" if sp_rmse > 30 else ""
    print(f"  sp{sp:2d}: {sp_rmse:6.2f}{marker}")

# ── 提出 ──────────────────────────────────────────────────────────────────────
OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
dtrain_f = lgb.Dataset(Xtr_full, label=y_pow)
final = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_full), 0, None) ** (1 / P_POWER)
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: GroupSeq+P1, LOSO={rmse:.4f}", loso=rmse)

print(f"\n[Done] {EXP}")
