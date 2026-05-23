"""
Experiment NA2: EPO + 波長選択 + PLS
=================================================================
NA1でCARS-PLSがEPOなしでLOSO=17.35を達成。
EPOで樹種変動を除去してから波長選択すれば更に改善する仮説。

パイプライン:
  MSC + SG(w=9,p=2) → EPO(n=5) → 波長選択 → PLS

比較:
  A) EPO + 全波長PLS    (ベースライン)
  B) EPO + CARS + PLS
  C) EPO + SiPLS
  D) EPO + VIP + PLS

参照: P1 LOSO=15.4725 (EPO+LGBM), NA1 CARS-PLS LOSO=17.3501
"""
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from itertools import combinations

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import load_data, msc, sg_deriv, loso_folds, loso_rmse, save_submission

import warnings; warnings.filterwarnings("ignore")

EXP = "NA2"

# ── データ・前処理 ────────────────────────────────────────────────────────────
data = load_data()
y = data["y_train"]
sp = data["sp_train"]
wns = data["wns"]

ref = data["X_train_raw"].mean(axis=0)
X_msc_sg_tr = sg_deriv(msc(data["X_train_raw"], ref), window=9, polyorder=2)
X_msc_sg_te = sg_deriv(msc(data["X_test_raw"],  ref), window=9, polyorder=2)


# ── EPO ───────────────────────────────────────────────────────────────────────
def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4:
            continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < 2:
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


V_epo = compute_epo_matrix(X_msc_sg_tr, y, sp, n_components=5)
X_tr = apply_epo(X_msc_sg_tr, V_epo)
X_te = apply_epo(X_msc_sg_te, V_epo)
print(f"EPO適用完了: {X_tr.shape}")


# ── ユーティリティ ────────────────────────────────────────────────────────────
def loso_pls(X, y, sp, n_comp):
    oof = np.zeros(len(y))
    for tr_idx, va_idx, _ in loso_folds(sp):
        nc = min(n_comp, X[tr_idx].shape[1] - 1, len(tr_idx) - 1)
        if nc < 1:
            oof[va_idx] = y.mean()
            continue
        pls = PLSRegression(n_components=nc)
        pls.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = pls.predict(X[va_idx]).ravel()
    return loso_rmse(np.clip(oof, 0, None), y), oof


def vip_scores(pls_model):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros(p)
    ss = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    for i in range(p):
        weight = (w[i, :] / np.linalg.norm(w, axis=0))**2
        vip[i] = np.sqrt(p * np.sum(ss * weight) / np.sum(ss))
    return vip


# ═══════════════════════════════════════════════════════════════════════════════
# A) EPO + 全波長 PLS (n_comp探索)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) EPO + 全波長 PLS")
print("="*60)

best_a_rmse = np.inf
best_a_nc = 5
for nc in [3, 5, 8, 10, 15, 20]:
    rmse, _ = loso_pls(X_tr, y, sp, nc)
    print(f"  n_comp={nc:>3}: LOSO={rmse:.4f}")
    if rmse < best_a_rmse:
        best_a_rmse = rmse
        best_a_nc = nc

print(f"\nEPO+全波長PLS ベスト: LOSO={best_a_rmse:.4f} (n_comp={best_a_nc})")


# ═══════════════════════════════════════════════════════════════════════════════
# B) EPO + CARS + PLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) EPO + CARS + PLS")
print("="*60)

def cars_select(X, y, sp, n_iter=50, n_comp=10, min_waves=5):
    n_samples, n_features = X.shape
    current_mask = np.ones(n_features, dtype=bool)

    wave_counts = np.round(
        np.exp(np.linspace(np.log(n_features), np.log(min_waves), n_iter))
    ).astype(int)
    wave_counts = np.clip(wave_counts, min_waves, n_features)

    history = []

    for i, n_keep in enumerate(wave_counts):
        X_sel = X[:, current_mask]
        nc = min(n_comp, X_sel.shape[1] - 1, n_samples - 1)
        if nc < 1:
            break

        pls = PLSRegression(n_components=nc)
        pls.fit(X_sel, y)
        coefs = np.abs(pls.coef_.ravel())

        sel_local = np.argsort(coefs)[::-1][:n_keep]
        orig_indices = np.where(current_mask)[0]
        new_mask = np.zeros(n_features, dtype=bool)
        new_mask[orig_indices[sel_local]] = True
        current_mask = new_mask

        if i % 5 == 0 or n_keep <= 30:
            rmse, _ = loso_pls(X[:, current_mask], y, sp, nc)
            history.append((n_keep, current_mask.copy(), rmse, nc))
            print(f"  iter={i+1:>3}, n_waves={n_keep:>4}, n_comp={nc}, LOSO={rmse:.4f}")

    best = min(history, key=lambda x: x[2])
    return best[1], best[2], best[3]

best_b_mask, best_b_rmse, best_b_nc = cars_select(X_tr, y, sp, n_iter=50, n_comp=15)
print(f"\nEPO+CARS-PLS ベスト: LOSO={best_b_rmse:.4f} (n_waves={best_b_mask.sum()}, n_comp={best_b_nc})")


# ═══════════════════════════════════════════════════════════════════════════════
# C) EPO + SiPLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) EPO + SiPLS")
print("="*60)

N_INTERVALS = 20
n_waves_total = X_tr.shape[1]
interval_size = n_waves_total // N_INTERVALS

intervals = []
for k in range(N_INTERVALS):
    start = k * interval_size
    end = start + interval_size if k < N_INTERVALS - 1 else n_waves_total
    intervals.append(np.arange(start, end))

print(f"\n{'区間':>4}  {'波数範囲(cm-1)':>18}  {'LOSO-RMSE':>10}")
print("-" * 40)

single_rmses = []
for k, idx in enumerate(intervals):
    X_int = X_tr[:, idx]
    rmse, _ = loso_pls(X_int, y, sp, n_comp=min(10, len(idx)-1))
    wn_range = f"{wns[idx[-1]]:.0f}-{wns[idx[0]]:.0f}"
    print(f"{k+1:>4}  {wn_range:>18}  {rmse:>10.4f}")
    single_rmses.append(rmse)

top5_idx = np.argsort(single_rmses)[:5]
print(f"\n上位5区間: {[k+1 for k in top5_idx]}")
print(f"\nシナジー探索:")
print(f"\n{'組み合わせ':>15}  {'選択波長数':>8}  {'LOSO-RMSE':>10}")
print("-" * 45)

best_c_rmse = min(single_rmses)
best_c_comb = [top5_idx[0]]

for r in [2, 3]:
    for comb in combinations(top5_idx, r):
        idx_all = np.concatenate([intervals[k] for k in comb])
        X_comb = X_tr[:, idx_all]
        nc = min(15, X_comb.shape[1] - 1)
        rmse, _ = loso_pls(X_comb, y, sp, nc)
        label = "+".join([str(k+1) for k in sorted(comb)])
        print(f"区間 [{label:>10}]  {len(idx_all):>8}  {rmse:>10.4f}")
        if rmse < best_c_rmse:
            best_c_rmse = rmse
            best_c_comb = list(comb)

si_best_idx = np.concatenate([intervals[k] for k in best_c_comb])
best_c_mask = np.zeros(n_waves_total, dtype=bool)
best_c_mask[si_best_idx] = True
print(f"\nEPO+SiPLS ベスト: LOSO={best_c_rmse:.4f} (区間={[k+1 for k in best_c_comb]}, n_waves={best_c_mask.sum()})")


# ═══════════════════════════════════════════════════════════════════════════════
# D) EPO + VIP + PLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D) EPO + VIP + PLS")
print("="*60)

pls_vip = PLSRegression(n_components=10)
pls_vip.fit(X_tr, y)
vip = vip_scores(pls_vip)

best_d_rmse = np.inf
best_d_mask = None
best_d_nc = 10

for thr in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    mask = vip >= thr
    if mask.sum() < 3:
        continue
    X_sel = X_tr[:, mask]
    for nc in [5, 8, 10, 15]:
        if nc >= X_sel.shape[1]:
            continue
        rmse, _ = loso_pls(X_sel, y, sp, nc)
        if rmse < best_d_rmse:
            best_d_rmse = rmse
            best_d_mask = mask
            best_d_nc = nc
    print(f"  VIP>={thr:.1f}: n_waves={mask.sum()}, LOSO={best_d_rmse:.4f}")

print(f"\nEPO+VIP-PLS ベスト: LOSO={best_d_rmse:.4f} (n_waves={best_d_mask.sum() if best_d_mask is not None else '-'})")


# ═══════════════════════════════════════════════════════════════════════════════
# サマリ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("サマリ")
print("="*60)
print(f"  参照 P1 (EPO+LGBM)     : LOSO=15.4725")
print(f"  参照 NA1 CARS-PLS      : LOSO=17.3501")
print(f"  A) EPO+全波長PLS       : LOSO={best_a_rmse:.4f}")
print(f"  B) EPO+CARS-PLS        : LOSO={best_b_rmse:.4f}  (n_waves={best_b_mask.sum()})")
print(f"  C) EPO+SiPLS           : LOSO={best_c_rmse:.4f}  (n_waves={best_c_mask.sum()})")
print(f"  D) EPO+VIP-PLS         : LOSO={best_d_rmse:.4f}  (n_waves={best_d_mask.sum() if best_d_mask is not None else '-'})")
print()

# 最良でsubmission生成
results = [
    ("EPO+CARS-PLS", best_b_rmse, best_b_mask, best_b_nc),
    ("EPO+SiPLS",    best_c_rmse, best_c_mask, 15),
    ("EPO+VIP-PLS",  best_d_rmse, best_d_mask, best_d_nc),
]
best_name, best_rmse_final, best_mask_final, best_nc_final = min(results, key=lambda x: x[1])

if best_mask_final is not None:
    print(f"最良: {best_name} (LOSO={best_rmse_final:.4f})")
    pls_final = PLSRegression(n_components=min(best_nc_final, best_mask_final.sum() - 1))
    pls_final.fit(X_tr[:, best_mask_final], y)
    preds = np.clip(pls_final.predict(X_te[:, best_mask_final]).ravel(), 0, None)
    out_path = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na2_best.csv"
    save_submission(data["test_ids"], preds, out_path)
