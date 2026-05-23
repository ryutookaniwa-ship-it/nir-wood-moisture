"""
Experiment L4: EPOをfold内で計算（LOSOリーク除去）
===================================================
現状のI2: EPOをtrain全体で計算 → validation種のスペクトルがEPO方向に混入
改善: 各LOSOフォールドのtrain分割のみでEPOを計算

これにより:
  1. CVが正直化（A3実験のMSCリーク修正と同じ原理）
  2. test時はtrain全体でEPO計算（本番は変わらず）
  3. LBスコアとLOSO-CVのギャップが縮小する可能性

パイプライン: MSC+SG(w=9,p=2)+[fold内EPO(n=5)]+sqrt(y)+LGBM(I2-params)
ベース: I2 (LOSO=15.73, LB=16.101)
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

EXP = "L4"

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

params = {
    **LGBM_BASE_PARAMS,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
}

ref = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)

print(f"=== Experiment {EXP}: EPOをfold内で計算（LOSOリーク除去） ===")
print(f"I2ベース(全体EPO, LOSO=15.73, LB=16.101)との比較\n")

# ── fold内EPO: 各foldのtrain分割でEPO計算 ─────────────────────────────────
oof_fold = np.zeros(len(y_train)); best_iters_fold = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    V_fold = compute_epo_matrix(Xtr_sg[tr_idx], y_train[tr_idx], sp_train[tr_idx], n_components=5)
    Xtr_f = apply_epo(Xtr_sg[tr_idx], V_fold)
    Xva_f = apply_epo(Xtr_sg[va_idx], V_fold)

    dtrain = lgb.Dataset(Xtr_f, label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(Xva_f, label=y_sqrt[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
    oof_fold[va_idx] = np.clip(m.predict(Xva_f), 0, None) ** 2
    best_iters_fold.append(m.best_iteration)

rmse_fold = loso_rmse(oof_fold, y_train)
avg_r_fold = int(np.mean(best_iters_fold))
print(f"fold内EPO:  LOSO={rmse_fold:.4f}  avg_iter={avg_r_fold}")
print(f"全体EPO(I2): LOSO=15.7282  avg_iter=496")
print(f"差分: {rmse_fold - 15.7282:+.4f}")

# ── 提出用: train全体でEPO計算（本番と同じ） ─────────────────────────────
V_full = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V_full)
Xte_epo = apply_epo(Xte_sg, V_full)

if rmse_fold < 15.73:
    dtrain_f = lgb.Dataset(Xtr_epo, label=y_sqrt)
    final = lgb.train(params, dtrain_f, num_boost_round=avg_r_fold,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte_epo), 0, None) ** 2
    OUT = rf"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_L4_epo_fold.csv"
    save_submission(test_ids, preds, OUT)
    print(f"\n[Done] {OUT}")
else:
    print("\n[Skip] fold内EPOはI2を超えなかった（提出はI2と同一なので不要）")
    print("注: 本番提出はtrain全体EPOなのでLBは変わらない。CV正直化のみの効果")
