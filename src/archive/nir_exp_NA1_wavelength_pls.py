"""
Experiment NA1: 波長選択 + PLS (P1から完全に離れたアプローチ)
=================================================================
P1 (EPO+LGBM) への固執をやめ、化学計量学的な正攻法を試す。

3手法を比較:
  A) VIP-PLS    : PLSのVIPスコアで波長選択 → PLS
  B) CARS-PLS   : Competitive Adaptive Reweighted Sampling → PLS
  C) SiPLS      : スペクトルを区間分割、最適区間組み合わせ → PLS

前処理: MSC + SG(w=9, p=2) のみ (EPOなし)
評価: LOSO-RMSE
"""
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import load_data, msc, sg_deriv, loso_folds, loso_rmse, save_submission

import warnings; warnings.filterwarnings("ignore")

EXP = "NA1"

# ── データ読み込み・前処理 ───────────────────────────────────────────────────
data = load_data()
y = data["y_train"]
sp = data["sp_train"]
wns = data["wns"]

ref = data["X_train_raw"].mean(axis=0)
X_tr = sg_deriv(msc(data["X_train_raw"], ref), window=9, polyorder=2)
X_te = sg_deriv(msc(data["X_test_raw"],  ref), window=9, polyorder=2)


# ── ユーティリティ ────────────────────────────────────────────────────────────
def loso_pls(X, y, sp, n_comp):
    """LOSO-CV で PLS を評価。oof と RMSE を返す。"""
    oof = np.zeros(len(y))
    for tr_idx, va_idx, _ in loso_folds(sp):
        pls = PLSRegression(n_components=min(n_comp, X[tr_idx].shape[1], len(tr_idx)-1))
        pls.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = pls.predict(X[va_idx]).ravel()
    return loso_rmse(np.clip(oof, 0, None), y), oof


def vip_scores(pls_model, X):
    """VIP (Variable Importance in Projection) スコアを計算。"""
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros(p)
    ss = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    for i in range(p):
        weight = (w[i, :] / np.linalg.norm(w, axis=0)) ** 2
        vip[i] = np.sqrt(p * np.sum(ss * weight) / np.sum(ss))
    return vip


# ═══════════════════════════════════════════════════════════════════════════════
# A) VIP-PLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) VIP-PLS")
print("="*60)

# まず全波長でPLSを fit してVIPスコア計算
n_comp_full = 10
pls_full = PLSRegression(n_components=n_comp_full)
pls_full.fit(X_tr, y)
vip = vip_scores(pls_full, X_tr)

# VIP閾値ごとに評価
print(f"\n{'VIP閾値':>8}  {'選択波長数':>8}  {'LOSO-RMSE':>10}")
print("-" * 35)

best_vip_rmse = np.inf
best_vip_mask = None
best_vip_ncomp = None

for thr in [0.5, 0.8, 1.0, 1.2, 1.5]:
    mask = vip >= thr
    if mask.sum() < 3:
        continue
    X_sel = X_tr[:, mask]
    best_rmse_nc = np.inf
    best_nc = 5
    for nc in [3, 5, 8, 10, 15]:
        if nc >= X_sel.shape[1]:
            continue
        rmse, _ = loso_pls(X_sel, y, sp, nc)
        if rmse < best_rmse_nc:
            best_rmse_nc = rmse
            best_nc = nc
    print(f"{thr:>8.1f}  {mask.sum():>8}  {best_rmse_nc:>10.4f}  (n_comp={best_nc})")
    if best_rmse_nc < best_vip_rmse:
        best_vip_rmse = best_rmse_nc
        best_vip_mask = mask
        best_vip_ncomp = best_nc

print(f"\nVIP-PLS ベスト LOSO-RMSE: {best_vip_rmse:.4f}  (n_waves={best_vip_mask.sum()}, n_comp={best_vip_ncomp})")


# ═══════════════════════════════════════════════════════════════════════════════
# B) CARS-PLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) CARS-PLS (Competitive Adaptive Reweighted Sampling)")
print("="*60)

def cars_select(X, y, n_iter=50, n_comp=10, min_waves=5):
    """
    CARS: 各iterationでPLS回帰係数の大きい上位波長を選択。
    波長数は指数的に減少。各subsetでLOSOを計算し最良subsetを返す。
    """
    n_samples, n_features = X.shape
    current_mask = np.ones(n_features, dtype=bool)

    # 指数的減少: iter 1→全波長, iter N→min_waves
    wave_counts = np.round(
        np.exp(np.linspace(np.log(n_features), np.log(min_waves), n_iter))
    ).astype(int)
    wave_counts = np.clip(wave_counts, min_waves, n_features)

    history = []  # (n_waves, mask, rmse)

    for i, n_keep in enumerate(wave_counts):
        X_sel = X[:, current_mask]
        nc = min(n_comp, X_sel.shape[1] - 1, n_samples - 1)
        if nc < 1:
            break

        # 全訓練データでPLS fit → |回帰係数|で順位付け
        pls = PLSRegression(n_components=nc)
        pls.fit(X_sel, y)
        coefs = np.abs(pls.coef_.ravel())

        # 上位 n_keep を選択
        sel_local = np.argsort(coefs)[::-1][:n_keep]
        orig_indices = np.where(current_mask)[0]
        new_mask = np.zeros(n_features, dtype=bool)
        new_mask[orig_indices[sel_local]] = True
        current_mask = new_mask

        # LOSO評価 (5 iterごとに評価して高速化)
        if i % 5 == 0 or n_keep <= 20:
            rmse, _ = loso_pls(X[:, current_mask], y, sp, nc)
            history.append((n_keep, current_mask.copy(), rmse))
            print(f"  iter={i+1:>3}, n_waves={n_keep:>4}, n_comp={nc}, LOSO={rmse:.4f}")

    # 最良
    best = min(history, key=lambda x: x[2])
    return best[1], best[2]

best_cars_mask, best_cars_rmse = cars_select(X_tr, y, n_iter=50, n_comp=10, min_waves=5)
print(f"\nCARS-PLS ベスト LOSO-RMSE: {best_cars_rmse:.4f}  (n_waves={best_cars_mask.sum()})")


# ═══════════════════════════════════════════════════════════════════════════════
# C) SiPLS (Synergy Interval PLS)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) SiPLS (Synergy Interval PLS)")
print("="*60)

N_INTERVALS = 20  # スペクトルを20区間に分割
n_waves = X_tr.shape[1]
interval_size = n_waves // N_INTERVALS

intervals = []
for k in range(N_INTERVALS):
    start = k * interval_size
    end = start + interval_size if k < N_INTERVALS - 1 else n_waves
    intervals.append(np.arange(start, end))

# 各区間のLOSO評価
print(f"\nスペクトル: {n_waves}波長 → {N_INTERVALS}区間 (各{interval_size}波長)")
print(f"\n{'区間':>4}  {'波数範囲(cm-1)':>18}  {'LOSO-RMSE':>10}")
print("-" * 40)

single_rmses = []
for k, idx in enumerate(intervals):
    X_int = X_tr[:, idx]
    rmse, _ = loso_pls(X_int, y, sp, n_comp=min(10, len(idx)-1))
    wn_range = f"{wns[idx[-1]]:.0f}-{wns[idx[0]]:.0f}"
    print(f"{k+1:>4}  {wn_range:>18}  {rmse:>10.4f}")
    single_rmses.append(rmse)

# 上位5区間を抽出してシナジー組み合わせ
top5_idx = np.argsort(single_rmses)[:5]
print(f"\n上位5区間: {[k+1 for k in top5_idx]}")
print(f"\nシナジー探索 (上位5区間の2〜3区間組み合わせ):")
print(f"\n{'組み合わせ':>15}  {'選択波長数':>8}  {'LOSO-RMSE':>10}")
print("-" * 45)

best_si_rmse = min(single_rmses)
best_si_comb = [top5_idx[0]]
best_si_mask = None

# 全5区間と組み合わせ探索
for r in [2, 3]:
    for comb in combinations(top5_idx, r):
        idx_all = np.concatenate([intervals[k] for k in comb])
        X_comb = X_tr[:, idx_all]
        nc = min(10, X_comb.shape[1] - 1)
        rmse, _ = loso_pls(X_comb, y, sp, nc)
        label = "+".join([str(k+1) for k in sorted(comb)])
        print(f"区間 [{label:>10}]  {len(idx_all):>8}  {rmse:>10.4f}")
        if rmse < best_si_rmse:
            best_si_rmse = rmse
            best_si_comb = list(comb)

si_best_idx = np.concatenate([intervals[k] for k in best_si_comb])
best_si_mask = np.zeros(n_waves, dtype=bool)
best_si_mask[si_best_idx] = True

print(f"\nSiPLS ベスト LOSO-RMSE: {best_si_rmse:.4f}  (区間={[k+1 for k in best_si_comb]}, n_waves={best_si_mask.sum()})")


# ═══════════════════════════════════════════════════════════════════════════════
# サマリ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("サマリ (P1ベスト: LOSO=15.4725)")
print("="*60)
print(f"  A) VIP-PLS  : LOSO={best_vip_rmse:.4f}  (n_waves={best_vip_mask.sum() if best_vip_mask is not None else '-'})")
print(f"  B) CARS-PLS : LOSO={best_cars_rmse:.4f}  (n_waves={best_cars_mask.sum()})")
print(f"  C) SiPLS    : LOSO={best_si_rmse:.4f}  (n_waves={best_si_mask.sum() if best_si_mask is not None else '-'})")
print()

# 最良手法でsubmission生成
results = [
    ("VIP-PLS",  best_vip_rmse,  best_vip_mask,  best_vip_ncomp),
    ("CARS-PLS", best_cars_rmse, best_cars_mask, 10),
    ("SiPLS",    best_si_rmse,   best_si_mask,   10),
]
best_name, best_rmse_final, best_mask_final, best_nc_final = min(results, key=lambda x: x[1])

if best_mask_final is not None:
    print(f"最良: {best_name} (LOSO={best_rmse_final:.4f})")
    # 全訓練データで再fit
    pls_final = PLSRegression(n_components=min(best_nc_final, best_mask_final.sum() - 1))
    pls_final.fit(X_tr[:, best_mask_final], y)
    preds = np.clip(pls_final.predict(X_te[:, best_mask_final]).ravel(), 0, None)

    out_path = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na1_best.csv"
    save_submission(data["test_ids"], preds, out_path)
