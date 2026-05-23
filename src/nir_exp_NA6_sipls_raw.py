"""
Experiment NA6: SiPLS (前処理なし/SNVのみ) — アプローチ3
=================================================================
P1の前処理(MSC+SG+EPO)に依存しない完全に独立したアプローチ。

比較:
  A) Raw + SiPLS + PLS
  B) SNV + SiPLS + PLS
  C) SNV + SG(w=11,p=2) + SiPLS + PLS  (EPOなし)
  D) SNV + SiPLS + LGBM(P1-params)
"""
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from itertools import combinations
import lightgbm as lgb

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, snv, sg_deriv, loso_folds, loso_rmse, save_submission
)
import warnings; warnings.filterwarnings("ignore")

EXP = "NA6"
N_INTERVALS = 20

data = load_data()
y = data["y_train"]; sp = data["sp_train"]; wns = data["wns"]
X_raw_tr = data["X_train_raw"]; X_raw_te = data["X_test_raw"]

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

def make_intervals(n_waves, n_int):
    sz = n_waves // n_int
    ivs = []
    for k in range(n_int):
        s = k * sz
        e = s + sz if k < n_int - 1 else n_waves
        ivs.append(np.arange(s, e))
    return ivs

def loso_pls(X, y, sp, n_comp=10):
    oof = np.zeros(len(y))
    for tr_idx, va_idx, _ in loso_folds(sp):
        nc = min(n_comp, X[tr_idx].shape[1]-1, len(tr_idx)-1)
        if nc < 1: oof[va_idx] = y.mean(); continue
        pls = PLSRegression(n_components=nc)
        pls.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = np.clip(pls.predict(X[va_idx]).ravel(), 0, None)
    return loso_rmse(oof, y), oof

def loso_lgbm_sqrt(X, y, sp):
    y_s = np.sqrt(y); oof = np.zeros(len(y)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=y_s[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=y_s[va_idx], reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(X[va_idx]), 0, None)**2
        iters.append(m.best_iteration)
    return loso_rmse(oof, y), int(np.mean(iters)), oof

def best_sipls(X_tr, X_te, y, sp, n_int=20, model="pls", top_k=5):
    ivs = make_intervals(X_tr.shape[1], n_int)
    single = []
    for k, idx in enumerate(ivs):
        if model == "pls":
            rmse, _ = loso_pls(X_tr[:, idx], y, sp)
        else:
            rmse, _, _ = loso_lgbm_sqrt(X_tr[:, idx], y, sp)
        single.append(rmse)

    print(f"\n  区間別LOSO ({model}):")
    for k, (idx, r) in enumerate(zip(ivs, single)):
        wn_lo, wn_hi = wns[idx[-1]], wns[idx[0]]
        print(f"    {k+1:>2}: {wn_lo:.0f}-{wn_hi:.0f} cm-1  LOSO={r:.4f}")

    top_idx = np.argsort(single)[:top_k]
    best_rmse = min(single); best_comb = [top_idx[0]]

    print(f"\n  シナジー探索 (上位{top_k}区間の組み合わせ):")
    for r_comb in [2, 3]:
        for comb in combinations(top_idx, r_comb):
            idx_all = np.concatenate([ivs[k] for k in comb])
            if model == "pls":
                rmse, _ = loso_pls(X_tr[:, idx_all], y, sp)
            else:
                rmse, _, _ = loso_lgbm_sqrt(X_tr[:, idx_all], y, sp)
            label = "+".join([str(k+1) for k in sorted(comb)])
            print(f"    [{label:>12}]  {len(idx_all):>5}波長  LOSO={rmse:.4f}")
            if rmse < best_rmse:
                best_rmse = rmse; best_comb = list(comb)

    best_mask = np.zeros(X_tr.shape[1], dtype=bool)
    best_mask[np.concatenate([ivs[k] for k in best_comb])] = True
    return best_rmse, best_mask, best_comb

print(f"=== Experiment {EXP}: SiPLS (前処理最小化) ===")
print(f"参照: P1 LOSO=15.4725\n")

results = []

# ── A) Raw + SiPLS + PLS ──────────────────────────────────────────────────────
print("="*55)
print("A) Raw + SiPLS + PLS")
rmse_a, mask_a, comb_a = best_sipls(X_raw_tr, X_raw_te, y, sp, model="pls")
print(f"\nA) LOSO={rmse_a:.4f}  区間={[k+1 for k in comb_a]}")
results.append(("A-Raw+SiPLS+PLS", rmse_a, mask_a, "raw", "pls"))

# ── B) SNV + SiPLS + PLS ─────────────────────────────────────────────────────
print("\n"+"="*55)
print("B) SNV + SiPLS + PLS")
X_snv_tr = snv(X_raw_tr); X_snv_te = snv(X_raw_te)
rmse_b, mask_b, comb_b = best_sipls(X_snv_tr, X_snv_te, y, sp, model="pls")
print(f"\nB) LOSO={rmse_b:.4f}  区間={[k+1 for k in comb_b]}")
results.append(("B-SNV+SiPLS+PLS", rmse_b, mask_b, "snv", "pls"))

# ── C) SNV+SG + SiPLS + PLS (EPOなし) ────────────────────────────────────────
print("\n"+"="*55)
print("C) SNV+SG(w=11,p=2) + SiPLS + PLS")
X_sg_tr = sg_deriv(X_snv_tr, window=11, polyorder=2)
X_sg_te = sg_deriv(X_snv_te, window=11, polyorder=2)
rmse_c, mask_c, comb_c = best_sipls(X_sg_tr, X_sg_te, y, sp, model="pls")
print(f"\nC) LOSO={rmse_c:.4f}  区間={[k+1 for k in comb_c]}")
results.append(("C-SNV+SG+SiPLS+PLS", rmse_c, mask_c, "sg", "pls"))

# ── D) SNV + SiPLS + LGBM ────────────────────────────────────────────────────
print("\n"+"="*55)
print("D) SNV + SiPLS + LGBM")
rmse_d, mask_d, comb_d = best_sipls(X_snv_tr, X_snv_te, y, sp, model="lgbm")
print(f"\nD) LOSO={rmse_d:.4f}  区間={[k+1 for k in comb_d]}")
results.append(("D-SNV+SiPLS+LGBM", rmse_d, mask_d, "snv", "lgbm"))

# ── サマリ ────────────────────────────────────────────────────────────────────
print("\n"+"="*55)
print("サマリ (参照: P1=15.4725)")
print("="*55)
for name, rmse, _, _, _ in results:
    mark = "✅" if rmse < 15.4725 else "❌"
    print(f"  {mark} {name}: LOSO={rmse:.4f}")

best = min(results, key=lambda x: x[1])
print(f"\nベスト: {best[0]}  LOSO={best[1]:.4f}")
