"""
Experiment NA3: 物理帯域 + EPO + LGBM(P1-params)
=================================================================
Y実験(物理帯域+LGBM, EPOなし, LOSO=21.78)のEPOあり版。

物理的根拠:
  - 5200 cm⁻¹ (combination band): 含水率との相関が最も高い
  - 6900 cm⁻¹ (1st overtone OH): 2番目に強い水吸収帯

2パターンで比較:
  A) EPO全波長後 → 物理帯域に絞る → LGBM
  B) 物理帯域に絞る → EPO → LGBM

帯域定義(広め/狭め 両方試す):
  広め: 4800-5600 cm⁻¹ + 6200-7400 cm⁻¹
  狭め: 5000-5400 cm⁻¹ + 6700-7200 cm⁻¹
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
import lightgbm as lgb

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse, save_submission
)
import warnings; warnings.filterwarnings("ignore")

EXP = "NA3"

# ── データ・前処理 ────────────────────────────────────────────────────────────
data = load_data()
y = data["y_train"]
sp = data["sp_train"]
wns = data["wns"]

ref = data["X_train_raw"].mean(axis=0)
X_sg_tr = sg_deriv(msc(data["X_train_raw"], ref), window=9, polyorder=2)
X_sg_te = sg_deriv(msc(data["X_test_raw"],  ref), window=9, polyorder=2)

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)


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


def loso_lgbm_sqrt(X, y, sp):
    y_sqrt = np.sqrt(y)
    oof = np.zeros(len(y)); best_iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(X[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)
    return loso_rmse(oof, y), int(np.mean(best_iters)), oof


# 物理帯域マスク定義
band_defs = {
    "wide":   [(4800, 5600), (6200, 7400)],
    "narrow": [(5000, 5400), (6700, 7200)],
    "combo":  [(4800, 5600)],       # combination band のみ
    "over":   [(6200, 7400)],       # overtone のみ
}

def make_band_mask(wns, bands):
    mask = np.zeros(len(wns), dtype=bool)
    for lo, hi in bands:
        mask |= (wns >= lo) & (wns <= hi)
    return mask


print(f"=== Experiment {EXP}: 物理帯域 + EPO + LGBM(P1-params) ===")
print(f"参照: P1 LOSO=15.4725, Y実験(帯域+LGBM, EPOなし) LOSO=21.78\n")
print(f"{'設定':>25}  {'波長数':>6}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 60)

results = []

# ── パターンA: EPO全波長 → 帯域絞り → LGBM ───────────────────────────────
V_full = compute_epo(X_sg_tr, y, sp, n=5)
X_epo_tr = apply_epo(X_sg_tr, V_full)
X_epo_te = apply_epo(X_sg_te, V_full)

for band_name, bands in band_defs.items():
    mask = make_band_mask(wns, bands)
    n_w = mask.sum()
    if n_w < 5: continue
    rmse, avg_i, oof = loso_lgbm_sqrt(X_epo_tr[:, mask], y, sp)
    label = f"A-{band_name}"
    print(f"{label:>25}  {n_w:>6}  {rmse:>8.4f}  {avg_i:>9}")
    results.append((label, rmse, mask, "A", oof))

# ── パターンB: 帯域絞り → EPO → LGBM ────────────────────────────────────
for band_name, bands in band_defs.items():
    mask = make_band_mask(wns, bands)
    n_w = mask.sum()
    if n_w < 5: continue
    X_band_tr = X_sg_tr[:, mask]
    X_band_te = X_sg_te[:, mask]
    V_band = compute_epo(X_band_tr, y, sp, n=5)
    X_bepo_tr = apply_epo(X_band_tr, V_band)
    X_bepo_te = apply_epo(X_band_te, V_band)
    rmse, avg_i, oof = loso_lgbm_sqrt(X_bepo_tr, y, sp)
    label = f"B-{band_name}"
    print(f"{label:>25}  {n_w:>6}  {rmse:>8.4f}  {avg_i:>9}")
    results.append((label, rmse, mask, "B", oof))

# ── ベスト ──────────────────────────────────────────────────────────────────
best = min(results, key=lambda x: x[1])
print(f"\nベスト: {best[0]}  LOSO={best[1]:.4f}")

# submission生成
best_label, best_rmse, best_mask, best_pat, _ = best
if best_pat == "A":
    X_te_final = X_epo_te[:, best_mask]
    X_tr_final = X_epo_tr[:, best_mask]
else:
    X_band_te_b = X_sg_te[:, best_mask]
    X_band_tr_b = X_sg_tr[:, best_mask]
    V_b = compute_epo(X_band_tr_b, y, sp, n=5)
    X_tr_final = apply_epo(X_band_tr_b, V_b)
    X_te_final = apply_epo(X_band_te_b, V_b)

y_sqrt = np.sqrt(y)
avg_iters = []
for tr_idx, va_idx, _ in loso_folds(sp):
    dtrain = lgb.Dataset(X_tr_final[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(X_tr_final[va_idx], label=y_sqrt[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    avg_iters.append(m.best_iteration)

avg_iter = int(np.mean(avg_iters))
final_model = lgb.train(P1_PARAMS, lgb.Dataset(X_tr_final, label=y_sqrt),
                        num_boost_round=avg_iter)
preds = np.clip(final_model.predict(X_te_final), 0, None) ** 2

out_path = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na3_best.csv"
save_submission(data["test_ids"], preds, out_path)
print(f"LOSO={best_rmse:.4f}, avg_iter={avg_iter}")
