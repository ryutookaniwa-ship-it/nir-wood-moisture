"""
Experiment M1: Huber損失への変更
=================================
現状: objective='regression'(L2) → sp15の残差^2が巨大 → モデル歪み
改善: Huber損失でsp15外れ値の影響を抑制

探索: huber_delta x (regression / huber / mape)
  - regression (L2) [I2ベースライン]
  - huber, delta: [2, 4, 6, 8, 12] (sqrt空間での閾値)
  - mape (相対誤差)

パイプライン: I2固定 (MSC+SG(w=9,p=2)+EPO(n=5)+sqrt(y))
             ただしsqrt変換後の残差に対してHuberが効くか検証
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

EXP = "M1"

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
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_sg, V)
Xte = apply_epo(Xte_sg, V)

base_params = {**LGBM_BASE_PARAMS,
               "learning_rate": 0.02, "num_leaves": 63,
               "feature_fraction": 0.07, "min_child_samples": 10}

print(f"=== Experiment {EXP}: Huber/MAE損失 vs L2 ===")
print(f"I2ベース(L2, LOSO=15.73, LB=16.101)\n")
print(f"{'objective':>12}  {'delta':>6}  {'LOSO':>8}  {'avg_iter':>9}")
print("-" * 45)

best_rmse = np.inf; best_params = None; best_bi = None

configs = [("regression", None)] + [("huber", d) for d in [2, 4, 6, 8, 12]] + [("mape", None)]

for obj, delta in configs:
    p = {**base_params, "objective": obj, "metric": obj if obj != "regression" else "rmse"}
    if delta is not None:
        p["huber_delta"] = delta

    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(p, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)

    rmse = loso_rmse(oof, y_train); avg_r = int(np.mean(best_iters))
    i2_tag = " [I2]" if obj == "regression" else ""
    flag   = " <-- best" if rmse < best_rmse else ""
    delta_str = f"{delta:6}" if delta is not None else "     -"
    print(f"  {obj:>12}  {delta_str}  {rmse:8.4f}  {avg_r:9d}{i2_tag}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_params = p; best_bi = best_iters

print(f"\nBest: {best_params['objective']}"
      f"{', delta='+str(best_params.get('huber_delta','')) if 'huber_delta' in best_params else ''}"
      f"  LOSO={best_rmse:.4f}")
print(f"vs I2(15.73): {best_rmse - 15.73:+.4f}")

if best_rmse < 15.73:
    dtrain_f = lgb.Dataset(Xtr, label=y_sqrt)
    final = lgb.train(best_params, dtrain_f, num_boost_round=int(np.mean(best_bi)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte), 0, None) ** 2
    obj_tag = best_params["objective"]
    delta_tag = f"_d{best_params.get('huber_delta','')}" if "huber_delta" in best_params else ""
    OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_M1_{obj_tag}{delta_tag}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"M1: I2+{obj_tag}{delta_tag} LOSO={best_rmse:.4f}", loso=best_rmse)
    print(f"\n[Done] {OUT}")
else:
    print("\n[Skip] I2を超えなかった")
