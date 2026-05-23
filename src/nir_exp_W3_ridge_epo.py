"""
Experiment W3: EPO後にRidge回帰（直接適用）
============================================
根拠: Discord参加者「Ridgeが安定して良いスコア」+ 我々はPLS+Ridgeは試したが
     「Ridgeのみ(PLSなし)」は未試行。

PLS+Ridge(E1)がLB=26.37で失敗した理由:
  PLS(15成分) が訓練樹種スペクトル空間の方向を抽出し、
  その方向がテスト樹種には適用不可

Ridge直接(W3)の期待:
  1555次元全特徴量にL2正則化を一様適用
  → 特定方向への集中がなく、樹種固有パターンを1成分に凝縮しない
  → LGBM(ff=0.07)と異なるメカニズムで汎化できる可能性

比較:
  - アルファグリッド: [0.01, 0.1, 1, 10, 100, 1000, 10000]
  - y^0.27変換あり/なしの両方

ベース: P1 (LOSO=15.4725, LB=15.395)
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")

from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP     = "W3"
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
data     = load_data()
y_train  = data["y_train"]
Xtr_raw  = data["X_train_raw"]
Xte_raw  = data["X_test_raw"]
test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref     = Xtr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(Xtr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(Xte_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr     = apply_epo(Xtr_sg, V)
Xte     = apply_epo(Xte_sg, V)

print(f"=== Experiment {EXP}: Ridge on EPO features ===")
print(f"Features: {Xtr.shape[1]}d  Samples: {Xtr.shape[0]}")

# ── Section 1: Ridge without y-transform ─────────────────────────────────────
print("\n--- Section 1: Ridge (no y-transform) ---")
print(f"{'alpha':>10}  {'LOSO':>8}  {'vs P1':>8}")
print("-" * 32)

alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
best_raw_rmse = np.inf; best_raw_alpha = None; best_raw_oof = None

for alpha in alphas:
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        m = Ridge(alpha=alpha)
        m.fit(Xtr[tr_idx], y_train[tr_idx])
        oof[va_idx] = m.predict(Xtr[va_idx])
    oof = np.clip(oof, 0, None)
    rmse = loso_rmse(oof, y_train)
    flag = " ←" if rmse < best_raw_rmse else ""
    print(f"  alpha={alpha:>8.2f}  {rmse:8.4f}  {rmse - P1_LOSO:+8.4f}{flag}")
    if rmse < best_raw_rmse:
        best_raw_rmse = rmse; best_raw_alpha = alpha; best_raw_oof = oof.copy()

print(f"\nBest (no transform): alpha={best_raw_alpha}  LOSO={best_raw_rmse:.4f}")

# ── Section 2: Ridge with y^0.27 transform ───────────────────────────────────
print("\n--- Section 2: Ridge (y^0.27) ---")
print(f"{'alpha':>10}  {'LOSO':>8}  {'vs P1':>8}")
print("-" * 32)

p = 0.27
y_trans = y_train ** p
best_trans_rmse = np.inf; best_trans_alpha = None; best_trans_oof = None

for alpha in alphas:
    oof_t = np.zeros(len(y_train))
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        m = Ridge(alpha=alpha)
        m.fit(Xtr[tr_idx], y_trans[tr_idx])
        oof_t[va_idx] = m.predict(Xtr[va_idx])
    oof = np.clip(oof_t, 0, None) ** (1.0 / p)
    rmse = loso_rmse(oof, y_train)
    flag = " ←" if rmse < best_trans_rmse else ""
    print(f"  alpha={alpha:>8.2f}  {rmse:8.4f}  {rmse - P1_LOSO:+8.4f}{flag}")
    if rmse < best_trans_rmse:
        best_trans_rmse = rmse; best_trans_alpha = alpha; best_trans_oof = oof.copy()

print(f"\nBest (y^0.27): alpha={best_trans_alpha}  LOSO={best_trans_rmse:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
best_overall = min(best_raw_rmse, best_trans_rmse)
print(f"\n=== Summary ===")
print(f"P1 (LGBM):         LOSO={P1_LOSO:.4f}")
print(f"W3 Ridge(raw):     LOSO={best_raw_rmse:.4f}  alpha={best_raw_alpha}")
print(f"W3 Ridge(y^0.27):  LOSO={best_trans_rmse:.4f}  alpha={best_trans_alpha}")

# Per-species breakdown for best Ridge config
if best_trans_rmse <= best_raw_rmse:
    best_oof = best_trans_oof; best_label = f"y^{p}"
else:
    best_oof = best_raw_oof; best_label = "raw"

print(f"\nPer-species RMSE (best Ridge, {best_label}):")
for sp in sorted(set(sp_train)):
    idx = np.where(sp_train == sp)[0]
    sp_rmse = np.sqrt(np.mean((y_train[idx] - best_oof[idx])**2))
    print(f"  sp{sp:02d}: {sp_rmse:6.2f}")

# ── Submit if improved ────────────────────────────────────────────────────────
if best_overall < P1_LOSO:
    delta = best_overall - P1_LOSO
    print(f"\n✅ P1 より {-delta:.4f} 改善 → 提出")
    if best_trans_rmse <= best_raw_rmse:
        alpha_use = best_trans_alpha; use_trans = True
        print(f"   使用: Ridge(alpha={alpha_use}, y^0.27)")
    else:
        alpha_use = best_raw_alpha; use_trans = False
        print(f"   使用: Ridge(alpha={alpha_use}, raw)")

    m_final = Ridge(alpha=alpha_use)
    if use_trans:
        m_final.fit(Xtr, y_train ** p)
        preds = np.clip(m_final.predict(Xte), 0, None) ** (1.0 / p)
    else:
        m_final.fit(Xtr, y_train)
        preds = np.clip(m_final.predict(Xte), 0, None)

    OUT = f"{OUT_DIR}/submission_{EXP}_alpha{alpha_use}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: Ridge(alpha={alpha_use}) LOSO={best_overall:.4f}",
                      loso=best_overall)
else:
    print(f"\n❌ P1比 {best_overall - P1_LOSO:+.4f} 悪化 → 提出なし")
    print("   → RidgeはLGBM(ff=0.07)より汎化性が低い")
