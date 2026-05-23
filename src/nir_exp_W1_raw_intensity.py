"""
Experiment W1: 生スペクトル強度特徴量の追加
=============================================
根拠: 主催者がQ&Aで「スペクトル全体の強度と含水率の強い正の相関」を物理的に確認済み。
     現パイプライン MSC → SG1deriv がこの情報を完全に破壊している。
       MSC: 全サンプルのゲイン/オフセットを統一 → 絶対強度を消去
       SG1: 微分 → ベースライン完全除去

施策: EPO特徴量(1555次元)に生スペクトル由来の5特徴量を追加。
  raw_mean      : 全波数の平均吸光度 (スペクトル全体の強度)
  raw_5187      : O-H combination band (最重要水吸収帯, 5187 cm⁻¹)
  raw_6896      : O-H 1st overtone    (重要水吸収帯, 6896 cm⁻¹)
  ratio_5187_4760: 水/セルロース比   (散乱非依存の化学量論的特徴)
  ratio_6896_5900: 水/C-H比          (散乱非依存の化学量論的特徴)

これらは各サンプル単体から計算するため LOSO リークなし。

ベース: P1 (LOSO=15.4725, LB=15.395)
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")

from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP     = "W1"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_LOSO = 15.4725

# ── EPO ──────────────────────────────────────────────────────────────────────
def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width; mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T

def apply_epo(X, V): return X - (X @ V) @ V.T

# ── Load & preprocess ─────────────────────────────────────────────────────────
data       = load_data()
y_train    = data["y_train"]
Xtr_raw    = data["X_train_raw"]
Xte_raw    = data["X_test_raw"]
test_ids   = data["test_ids"]
sp_train   = data["sp_train"]
wns        = data["wns"]

ref     = Xtr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(Xtr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(Xte_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── Raw intensity features ────────────────────────────────────────────────────
idx_5187 = int(np.argmin(np.abs(wns - 5187)))
idx_6896 = int(np.argmin(np.abs(wns - 6896)))
idx_4760 = int(np.argmin(np.abs(wns - 4760)))
idx_5900 = int(np.argmin(np.abs(wns - 5900)))

print(f"=== Experiment {EXP}: 生スペクトル強度特徴量の追加 ===")
print(f"Nearest wavenumbers: 5187→{wns[idx_5187]:.1f}, 6896→{wns[idx_6896]:.1f}, "
      f"4760→{wns[idx_4760]:.1f}, 5900→{wns[idx_5900]:.1f}")

def make_raw_feats(X_raw):
    mean  = X_raw.mean(axis=1)
    f5187 = X_raw[:, idx_5187]
    f6896 = X_raw[:, idx_6896]
    r1    = f5187 / (X_raw[:, idx_4760] + 1e-8)   # water / cellulose
    r2    = f6896 / (X_raw[:, idx_5900] + 1e-8)   # water / C-H
    return np.column_stack([mean, f5187, f6896, r1, r2])

raw_tr = make_raw_feats(Xtr_raw)
raw_te = make_raw_feats(Xte_raw)

print(f"\nRaw features: mean={raw_tr[:, 0].mean():.4f}±{raw_tr[:, 0].std():.4f}, "
      f"5187={raw_tr[:, 1].mean():.4f}±{raw_tr[:, 1].std():.4f}, "
      f"6896={raw_tr[:, 2].mean():.4f}±{raw_tr[:, 2].std():.4f}")

# Correlation with y_train
for i, name in enumerate(["raw_mean", "raw_5187", "raw_6896", "ratio_5187/4760", "ratio_6896/5900"]):
    r = np.corrcoef(raw_tr[:, i], y_train)[0, 1]
    print(f"  corr({name}, y) = {r:+.4f}")

# Concatenate
Xtr = np.hstack([Xtr_epo, raw_tr])   # 1555 + 5 = 1560 features
Xte = np.hstack([Xte_epo, raw_te])

# ── LGBM params (I2/P1 settings) ─────────────────────────────────────────────
params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

p = 0.27

# ── LOSO-CV ───────────────────────────────────────────────────────────────────
print(f"\n--- LOSO-CV (y^{p:.2f}) ---")
y_trans    = y_train ** p
oof_trans  = np.zeros(len(y_train))
iters      = []

for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xtr[va_idx])
    iters.append(m.best_iteration)

oof       = np.clip(oof_trans, 0, None) ** (1.0 / p)
rmse      = loso_rmse(oof, y_train)
avg_iter  = int(np.mean(iters))
delta     = rmse - P1_LOSO

print(f"\nW1  LOSO={rmse:.4f}  avg_iter={avg_iter}  vs P1: {delta:+.4f}")

# Per-species breakdown
print("\nPer-species RMSE:")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    sp_rmse = np.sqrt(np.mean((y_train[idx] - oof[idx])**2))
    print(f"  sp{sp:02d}: {sp_rmse:6.2f}")

# ── Submit if improved ────────────────────────────────────────────────────────
if rmse < P1_LOSO:
    print(f"\n✅ P1 より {-delta:.4f} 改善 → 提出")
    y_full = y_train ** p
    dtrain_f = lgb.Dataset(Xtr, label=y_full)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=avg_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte), 0, None) ** (1.0 / p)
    OUT = f"{OUT_DIR}/submission_{EXP}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: +raw_intensity LOSO={rmse:.4f}", loso=rmse)
else:
    print(f"\n❌ P1比 {delta:+.4f} 悪化 → 提出なし")
