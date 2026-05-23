"""
Experiment M2: ターゲット変換 grid search (I2パイプライン)
===========================================================
O実験(EPO前)ではsqrt(p=0.50)が最良。EPO後は最適変換が変わりうる。
特にlog1p変換はMC=298%の外挿幅を大きく縮小する。

  変換         MC=10  MC=100  MC=298  高MC圧縮比
  sqrt(p=0.5)   3.16   10.0   17.3    5.5x  [I2ベース]
  log1p         2.40    4.62   5.70    2.4x
  p=0.40        1.58    6.31  11.9    7.5x
  p=0.30        2.00    3.98   5.97   3.0x

探索: [log1p, p=0.20, p=0.30, p=0.40, p=0.50(sqrt), p=0.60]
パイプライン: I2固定 (MSC+SG(w=9,p=2)+EPO(n=5)+LGBM(I2-params))
ベース: I2 (LOSO=15.73, LB=16.101)
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "M2"

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

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

print(f"=== Experiment {EXP}: ターゲット変換 grid search (I2パイプライン) ===")
print(f"I2ベース(sqrt=p0.50, LOSO=15.73, LB=16.101)\n")
print(f"  MC=10: sqrt={10**0.5:.2f}, log1p={np.log1p(10):.2f}")
print(f"  MC=100: sqrt={100**0.5:.2f}, log1p={np.log1p(100):.2f}")
print(f"  MC=298: sqrt={298**0.5:.2f}, log1p={np.log1p(298):.2f}\n")
print(f"{'transform':>10}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 34)

best_rmse = np.inf; best_name = None; best_bi = None
best_Xte_preds = None

# 変換の定義: (名前, forward, inverse)
transforms = [
    ("log1p",  lambda y: np.log1p(y),      lambda p: np.expm1(np.clip(p, 0, None))),
    ("p=0.20", lambda y: y ** 0.20,         lambda p: np.clip(p, 0, None) ** (1/0.20)),
    ("p=0.30", lambda y: y ** 0.30,         lambda p: np.clip(p, 0, None) ** (1/0.30)),
    ("p=0.40", lambda y: y ** 0.40,         lambda p: np.clip(p, 0, None) ** (1/0.40)),
    ("p=0.50", lambda y: y ** 0.50,         lambda p: np.clip(p, 0, None) ** 2.0),   # I2
    ("p=0.60", lambda y: y ** 0.60,         lambda p: np.clip(p, 0, None) ** (1/0.60)),
]

for name, fwd, inv in transforms:
    y_trans = fwd(y_train)

    oof_trans = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        best_iters.append(m.best_iteration)

    oof = inv(oof_trans)
    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    i2_tag = " [I2]" if name == "p=0.50" else ""
    flag   = " <-- best" if rmse < best_rmse else ""
    print(f"  {name:>10}  {rmse:8.4f}  {avg_r:9d}{i2_tag}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_name = name; best_bi = best_iters
        _, inv_best = fwd, inv
        best_fwd, best_inv = fwd, inv

print(f"\nBest: {best_name}  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

if best_rmse < 15.73:
    y_trans_full = best_fwd(y_train)
    dtrain_f = lgb.Dataset(Xtr, label=y_trans_full)
    final = lgb.train(params, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = best_inv(final.predict(Xte))
    tag = best_name.replace("=", "").replace(".", "")
    OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_M2_{tag}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"M2: I2+{best_name} LOSO={best_rmse:.4f}", loso=best_rmse)
    print(f"\n[Done] {OUT}")
else:
    print("\n[Skip] I2を超えなかった")
