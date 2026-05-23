"""
Experiment J4: SG window/poly 細粒度探索
=================================================
I2(SG w=9,p=2)周辺のさらに細かいSGパラメータ探索。
EPO n=5固定、LGBM(B2-params)固定。

探索グリッド:
  (w=9, p=2) [I2]
  (w=9, p=3)
  (w=11, p=2)
  (w=11, p=3) [B2のw=5,p=3との中間]
  (w=13, p=2)
  (w=13, p=3)
  (w=7, p=2)
  (w=5, p=2) [p=3比較]
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "J4"

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
y_sqrt = np.sqrt(y_train)

params = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

sg_configs = [
    (5, 2), (5, 3),   # B2系のpoly=3との比較
    (7, 2), (7, 3),
    (9, 2), (9, 3),   # I2(w=9,p=2)周辺
    (11, 2), (11, 3),
    (13, 2), (13, 3),
]

print(f"=== Experiment {EXP}: SG細粒度探索 ===")
print(f"I2(w=9,p=2, LOSO=15.73) / B2(w=5,p=3, LOSO=16.44)との比較\n")
print(f"{'(w, p)':>10}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 35)

best_rmse = np.inf; best_w = None; best_p = None
best_Xtr = None; best_Xte = None; best_bi = None

for w, p in sg_configs:
    Xtr_pp = sg_deriv(msc(X_train_raw, ref), window=w, polyorder=p)
    Xte_pp = sg_deriv(msc(X_test_raw,  ref), window=w, polyorder=p)
    V = compute_epo_matrix(Xtr_pp, y_train, sp_train, n_components=5)
    Xtr_epo = apply_epo(Xtr_pp, V); Xte_epo = apply_epo(Xte_pp, V)

    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    tag = ""
    if w == 9 and p == 2: tag = " [I2]"
    if w == 5 and p == 3: tag = " [B2]"
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  (w={w:2d},p={p})  {rmse:8.4f}  {avg_r:9d}{tag}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_w = w; best_p = p
        best_Xtr = Xtr_epo.copy(); best_Xte = Xte_epo.copy(); best_bi = best_iters

print(f"\nBest: (w={best_w},p={best_p})  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

dtrain_f = lgb.Dataset(best_Xtr, label=y_sqrt)
final = lgb.train(params, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(best_Xte), 0, None) ** 2
OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_J4_sg_w{best_w}p{best_p}.csv"
save_submission(test_ids, preds, OUT)
print(f"\n[Done] {OUT}")
