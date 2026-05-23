"""
Experiment I3: EPO後追加正規化
=================================================
B2: MSC+SG+EPO(n=5) → LGBM
EPO後に残った樹種間スケール差をSNV/MSCで再補正する。

パイプライン候補:
  A: MSC → SG → EPO → SNV → LGBM
  B: MSC → SG → EPO → MSC(再適用) → LGBM
  C: MSC → SG → EPO → StandardScaler → LGBM (特徴量単位で標準化)
  D: B2のまま (ベースライン確認)

ベース: B2 (LOSO=16.44, LB=17.651)
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, snv, loso_folds, loso_rmse,
    save_submission, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "I3"

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
y_sqrt = np.sqrt(y_train)

ref = X_train_raw.mean(axis=0)
Xtr_pp = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
Xte_pp = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)
V = compute_epo_matrix(Xtr_pp, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_pp, V); Xte_epo = apply_epo(Xte_pp, V)

params = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

configs = {
    "D_B2baseline": (Xtr_epo, Xte_epo),
    "A_EPO+SNV":    (snv(Xtr_epo), snv(Xte_epo)),
    "B_EPO+MSC":    (lambda: (lambda ref2: (msc(Xtr_epo, ref2), msc(Xte_epo, ref2)))(Xtr_epo.mean(axis=0)))(),
}
sc = StandardScaler().fit(Xtr_epo)
configs["C_EPO+StdScaler"] = (sc.transform(Xtr_epo), sc.transform(Xte_epo))

print(f"=== Experiment {EXP}: EPO後追加正規化 ===")
print(f"B2(LOSO=16.44)との比較\n")
print(f"{'config':>18}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 40)

best_rmse = np.inf; best_name = None; best_Xtr = None; best_Xte = None; best_bi = None

for name, (Xtr, Xte) in configs.items():
    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  {name:>18}  {rmse:8.4f}  {avg_r:9d}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_name = name
        best_Xtr = Xtr; best_Xte = Xte; best_bi = best_iters

print(f"\nBest: {best_name}  LOSO={best_rmse:.4f}")
print(f"vs B2(16.44): {best_rmse - 16.44:+.4f}")

dtrain_f = lgb.Dataset(best_Xtr, label=y_sqrt)
final = lgb.train(params, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(best_Xte), 0, None) ** 2
OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_I3_post_epo_norm.csv"
save_submission(test_ids, preds, OUT)
print(f"\n[Done] {OUT}")
