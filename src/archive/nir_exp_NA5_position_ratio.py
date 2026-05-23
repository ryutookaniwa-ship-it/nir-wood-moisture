"""
Experiment NA5: position_ratio特徴量 + EPO + LGBM(P1-params)
=================================================================
仮説: テスト種内での測定順位(乾燥過程の位置)が含水率の強力な予測因子

position_ratio = (sample_number - 種内min) / (種内max - 種内min)
  → 0: 種内の最初の測定(飽水) → 1: 種内の最後の測定(乾燥)

各サンプル単体のメタデータ(sample_number + species_number)から計算可能。
他サンプルのスペクトル値や水分値は不要。

比較:
  A) position_ratio 単体
  B) P1スペクトル特徴量 + position_ratio
  C) EPO後スペクトル + position_ratio + p=0.27変換
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, TRAIN_PATH, TEST_PATH
)
import warnings; warnings.filterwarnings("ignore")

EXP = "NA5"

# ── データ読み込み ────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")

data = load_data()
y        = data["y_train"]
sp_tr    = data["sp_train"]
test_ids = data["test_ids"]

sn_tr  = train["sample number"].values
sn_te  = test["sample number"].values
sp_te  = test["species number"].values

# ── position_ratio 計算 ───────────────────────────────────────────────────────
def compute_position_ratio(sample_numbers, species):
    pos = np.zeros(len(sample_numbers))
    for sp in np.unique(species):
        mask = species == sp
        sn = sample_numbers[mask]
        mn, mx = sn.min(), sn.max()
        if mx == mn:
            pos[mask] = 0.5
        else:
            pos[mask] = (sn - mn) / (mx - mn)
    return pos

pos_tr = compute_position_ratio(sn_tr, sp_tr)
pos_te = compute_position_ratio(sn_te, sp_te)

print(f"=== Experiment {EXP}: position_ratio + EPO + LGBM ===")
print(f"参照: P1 LOSO=15.4725")
print(f"\nposition_ratio分布(train): min={pos_tr.min():.3f}, max={pos_tr.max():.3f}, mean={pos_tr.mean():.3f}")
print(f"position_ratio分布(test):  min={pos_te.min():.3f}, max={pos_te.max():.3f}, mean={pos_te.mean():.3f}")

# moisture vs position_ratio の相関確認
corr = np.corrcoef(pos_tr, y)[0, 1]
print(f"\nposition_ratio vs 含水率 相関: {corr:.4f}")

# ── 前処理 ───────────────────────────────────────────────────────────────────
ref = data["X_train_raw"].mean(axis=0)
X_sg_tr = sg_deriv(msc(data["X_train_raw"], ref), window=9, polyorder=2)
X_sg_te = sg_deriv(msc(data["X_test_raw"],  ref), window=9, polyorder=2)

def compute_epo(X, y, sp, n=5, bw=10.0):
    bins = np.arange(0, y.max() + bw, bw)
    dirs = []
    for lo in bins[:-1]:
        mask = (y >= lo) & (y < lo + bw)
        if mask.sum() < 4: continue
        sp_u = np.unique(sp[mask])
        if len(sp_u) < 2: continue
        means = np.array([X[mask][sp[mask] == s].mean(0) for s in sp_u])
        inter = means - means.mean(0)
        nc = min(n, inter.shape[0] - 1)
        if nc < 1: continue
        pca = PCA(n_components=nc, random_state=42).fit(inter)
        dirs.append(pca.components_)
    if not dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n].T

def apply_epo(X, V): return X - (X @ V) @ V.T

V = compute_epo(X_sg_tr, y, sp_tr, n=5)
X_epo_tr = apply_epo(X_sg_tr, V)
X_epo_te = apply_epo(X_sg_te, V)

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

def loso_lgbm(X, label, sp, transform=None, inv_transform=None):
    oof = np.zeros(len(y)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=label[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=label[va_idx], reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        pred = m.predict(X[va_idx])
        oof[va_idx] = inv_transform(pred) if inv_transform else pred
        iters.append(m.best_iteration)
    rmse = loso_rmse(np.clip(oof, 0, None), y)
    return rmse, int(np.mean(iters)), oof


print(f"\n{'設定':>35}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 60)

results = []

# ── A) position_ratio 単体 ────────────────────────────────────────────────────
X_a = pos_tr.reshape(-1, 1)
X_a_te = pos_te.reshape(-1, 1)
rmse_a, avg_a, _ = loso_lgbm(X_a, y, sp_tr)
print(f"{'A) position_ratio単体':>35}  {rmse_a:>8.4f}  {avg_a:>9}")
results.append(("A-pos_only", rmse_a, X_a, X_a_te, y, None, None))

# ── B) EPOスペクトル + position_ratio (y=raw) ─────────────────────────────────
X_b_tr = np.column_stack([X_epo_tr, pos_tr])
X_b_te = np.column_stack([X_epo_te, pos_te])
rmse_b, avg_b, _ = loso_lgbm(X_b_tr, y, sp_tr)
print(f"{'B) EPO+pos_ratio (raw y)':>35}  {rmse_b:>8.4f}  {avg_b:>9}")
results.append(("B-epo+pos", rmse_b, X_b_tr, X_b_te, y, None, None))

# ── C) EPOスペクトル + position_ratio + y^0.27 ────────────────────────────────
y_p = y ** 0.27
X_c_tr = X_b_tr; X_c_te = X_b_te
rmse_c, avg_c, _ = loso_lgbm(X_c_tr, y_p, sp_tr,
                               inv_transform=lambda p: np.clip(p, 0, None) ** (1/0.27))
print(f"{'C) EPO+pos_ratio+y^0.27':>35}  {rmse_c:>8.4f}  {avg_c:>9}")
results.append(("C-epo+pos+p27", rmse_c, X_c_tr, X_c_te, y_p,
                lambda p: np.clip(p,0,None)**(1/0.27), 0.27))

# ── D) EPOスペクトル + position_ratio + y^0.5 ─────────────────────────────────
y_s = np.sqrt(y)
rmse_d, avg_d, _ = loso_lgbm(X_c_tr, y_s, sp_tr,
                               inv_transform=lambda p: np.clip(p, 0, None) ** 2)
print(f"{'D) EPO+pos_ratio+sqrt(y)':>35}  {rmse_d:>8.4f}  {avg_d:>9}")
results.append(("D-epo+pos+sqrt", rmse_d, X_c_tr, X_c_te, y_s,
                lambda p: np.clip(p,0,None)**2, 0.5))

# ── E) position_ratio のみ (線形回帰で確認) ───────────────────────────────────
from sklearn.linear_model import Ridge
oof_e = np.zeros(len(y))
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    m = Ridge(alpha=1.0)
    m.fit(pos_tr[tr_idx].reshape(-1,1), y[tr_idx])
    oof_e[va_idx] = m.predict(pos_tr[va_idx].reshape(-1,1)).ravel()
rmse_e = loso_rmse(np.clip(oof_e, 0, None), y)
print(f"{'E) pos_ratio単体(Ridge)':>35}  {rmse_e:>8.4f}  {'—':>9}")

# ── サマリ ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"サマリ (参照: P1=15.4725)")
print(f"{'='*60}")
print(f"  A) pos_ratio単体(LGBM)  : LOSO={rmse_a:.4f}")
print(f"  B) EPO+pos+raw y        : LOSO={rmse_b:.4f}")
print(f"  C) EPO+pos+y^0.27       : LOSO={rmse_c:.4f}")
print(f"  D) EPO+pos+sqrt(y)      : LOSO={rmse_d:.4f}")
print(f"  E) pos_ratio単体(Ridge) : LOSO={rmse_e:.4f}")

best = min(results, key=lambda x: x[1])
print(f"\nベスト: {best[0]}  LOSO={best[1]:.4f}")

# ── submission生成 ────────────────────────────────────────────────────────────
bname, brmse, bXtr, bXte, blabel, binv, bpow = best
avg_iters = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    dtrain = lgb.Dataset(bXtr[tr_idx], label=blabel[tr_idx])
    dval   = lgb.Dataset(bXtr[va_idx], label=blabel[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    avg_iters.append(m.best_iteration)
avg_iter = int(np.mean(avg_iters))

final = lgb.train(P1_PARAMS, lgb.Dataset(bXtr, label=blabel), num_boost_round=avg_iter)
raw_preds = final.predict(bXte)
preds = binv(raw_preds) if binv else np.clip(raw_preds, 0, None)
preds = np.clip(preds, 0, None)

out = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na5_best.csv"
save_submission(test_ids, preds, out)
print(f"avg_iter={avg_iter}, LOSO={brmse:.4f}")
