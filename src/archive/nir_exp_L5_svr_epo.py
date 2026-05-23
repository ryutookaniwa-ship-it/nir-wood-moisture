"""
Experiment L5: SVR(RBF) on I2パイプライン
==========================================
EPO後の特徴量にSVRを適用。
過去のD実験(PLS+SVR, C=1000→LOSO=37.76)はEPOなし。
EPOで樹種方向を除去した後のSVRは未試験。

探索: C x gamma
  C:     [1, 10, 100]
  gamma: ['scale', 0.001, 0.0001]
  kernel: RBF固定

パイプライン: MSC+SG(w=9,p=2)+EPO(n=5)+StandardScaler+SVR(RBF)
※ターゲットはsqrt変換なし（SVRは変換不要）

ベース: I2 (LOSO=15.73, LB=16.101)
"""
import sys
import numpy as np
from itertools import product
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission,
)
import warnings; warnings.filterwarnings("ignore")

EXP = "L5"

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

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_sg, V)
Xte = apply_epo(Xte_sg, V)

print(f"=== Experiment {EXP}: SVR(RBF) on I2パイプライン ===")
print(f"I2ベース(LGBM, LOSO=15.73, LB=16.101)との比較\n")
print(f"{'C':>6}  {'gamma':>8}  {'LOSO':>8}")
print("-" * 32)

best_rmse = np.inf; best_cfg = None

for C, gamma in product([1, 10, 100], ['scale', 0.001, 0.0001]):
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr[tr_idx])
        Xva_s = scaler.transform(Xtr[va_idx])
        svr = SVR(kernel='rbf', C=C, gamma=gamma)
        svr.fit(Xtr_s, y_train[tr_idx])
        oof[va_idx] = np.clip(svr.predict(Xva_s), 0, None)

    rmse = loso_rmse(oof, y_train)
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  {C:5}  {str(gamma):>8}  {rmse:8.4f}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = (C, gamma)

bC, bg = best_cfg
print(f"\nBest: C={bC}, gamma={bg}  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

if best_rmse < 15.73:
    scaler_f = StandardScaler()
    Xtr_sf = scaler_f.fit_transform(Xtr)
    Xte_sf = scaler_f.transform(Xte)
    svr_f = SVR(kernel='rbf', C=bC, gamma=bg)
    svr_f.fit(Xtr_sf, y_train)
    preds = np.clip(svr_f.predict(Xte_sf), 0, None)
    OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_L5_svr_C{bC}.csv"
    save_submission(test_ids, preds, OUT)
    print(f"\n[Done] {OUT}")
else:
    print("\n[Skip] SVRはI2(LGBM)を超えなかった")
