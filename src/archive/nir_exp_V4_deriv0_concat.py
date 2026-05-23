"""
Experiment V4: 0次スペクトル（微分なし）＋1次微分のEPO結合
=============================================================
MSC後の0次スペクトル（吸収帯の強度情報）と
1次微分スペクトル（ベースライン除去・ピーク鮮明化）は互いに補完的。
両方をEPOしてhstackすると改善するか検証。

試すバリアント:
  V4a: EPO(SG1次) + EPO(0次)  concat 3110次元
  V4b: EPO(SG1次) + MSC(0次)  concat（0次はEPOなし）
  V4c: EPO(SG1次) + EPO(SG2次) concat（1次+2次微分）
  V4d: EPO(0次)のみ（SG微分なし）

ベース: P1 LOSO=15.4725, LB=15.395
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

EXP = "V4"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_LOSO = 15.4725


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
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

ref = X_train_raw.mean(axis=0)
Xtr_msc = msc(X_train_raw, ref)
Xte_msc = msc(X_test_raw,  ref)

# 0次（微分なし、MSCのみ）
Xtr_d0 = Xtr_msc.copy()
Xte_d0 = Xte_msc.copy()

# 1次微分 (P1ベース, w=9, poly=2)
Xtr_d1 = sg_deriv(Xtr_msc, window=9, polyorder=2, deriv=1)
Xte_d1 = sg_deriv(Xte_msc, window=9, polyorder=2, deriv=1)

# 2次微分
Xtr_d2 = sg_deriv(Xtr_msc, window=9, polyorder=2, deriv=2)
Xte_d2 = sg_deriv(Xte_msc, window=9, polyorder=2, deriv=2)

# EPO行列を各次数で計算
V_d1 = compute_epo_matrix(Xtr_d1, y_train, sp_train, n_components=5)
V_d0 = compute_epo_matrix(Xtr_d0, y_train, sp_train, n_components=5)
V_d2 = compute_epo_matrix(Xtr_d2, y_train, sp_train, n_components=5)

Xtr_epo_d1 = apply_epo(Xtr_d1, V_d1)
Xte_epo_d1 = apply_epo(Xte_d1, V_d1)
Xtr_epo_d0 = apply_epo(Xtr_d0, V_d0)
Xte_epo_d0 = apply_epo(Xte_d0, V_d0)
Xtr_epo_d2 = apply_epo(Xtr_d2, V_d2)
Xte_epo_d2 = apply_epo(Xte_d2, V_d2)

variants = {
    "V4a": (np.hstack([Xtr_epo_d1, Xtr_epo_d0]),
            np.hstack([Xte_epo_d1, Xte_epo_d0]),
            "EPO(SG1次) + EPO(0次)  3110次元"),
    "V4b": (np.hstack([Xtr_epo_d1, Xtr_d0]),
            np.hstack([Xte_epo_d1, Xte_d0]),
            "EPO(SG1次) + MSC(0次)  3110次元"),
    "V4c": (np.hstack([Xtr_epo_d1, Xtr_epo_d2]),
            np.hstack([Xte_epo_d1, Xte_epo_d2]),
            "EPO(SG1次) + EPO(SG2次) 3110次元"),
    "V4d": (Xtr_epo_d0, Xte_epo_d0,
            "EPO(0次)のみ            1555次元"),
}

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

p = 0.27
y_trans = y_train ** p
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / p)

print(f"=== Experiment {EXP}: 0次スペクトル+1次微分 結合 ===")
print(f"ベース P1: LOSO={P1_LOSO}\n")
print(f"{'variant':<6}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}  説明")
print("-" * 72)

best_rmse = P1_LOSO
best_key  = None
best_data = None

for key, (Xtr, Xte, desc) in variants.items():
    oof_trans = np.zeros(len(y_trans))
    iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)
    oof  = inv(oof_trans)
    rmse = loso_rmse(oof, y_train)
    avg_r = int(np.mean(iters))
    diff = rmse - P1_LOSO
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  {key:<6}  {rmse:8.4f}  {avg_r:9d}  {diff:+7.4f}  {desc}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_key  = key
        best_data = (Xtr, Xte, avg_r)

print()
if best_key:
    print(f"Best: {best_key}  LOSO={best_rmse:.4f}  vs P1: {best_rmse - P1_LOSO:+.4f}")
    Xtr_b, Xte_b, avg_r_b = best_data
    dtrain_f = lgb.Dataset(Xtr_b, label=y_train ** p)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=avg_r_b,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte_b))
    OUT = f"{OUT_DIR}/submission_{best_key}_deriv0.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{best_key}: deriv0 concat LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print(f"全バリアントがP1(LOSO={P1_LOSO})を超えず -> 提出なし")
