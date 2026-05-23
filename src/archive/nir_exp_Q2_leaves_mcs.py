"""
Experiment Q2: leaves / min_child_samples 再チューニング (p=0.27)
==================================================================
leaves=63, mcs=10 は p=0.50(sqrt) で確認済み (L1実験)。
p=0.27 変換後はターゲット分布が変わるため最適値が異なる可能性。
L1: leaves=31,mcs=30 → LOSO改善もLB悪化 → leaves=63,mcs=10 維持
今回は p=0.27 ベースで再探索。

探索:
  num_leaves: [31, 47, 63, 95, 127]
  min_child_samples: [5, 10, 20, 30]
ベース: P1 (LOSO=15.4725, LB=15.395)
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

EXP = "Q2"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
POWER = 0.27

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

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

y_trans = y_train ** POWER
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / POWER)

base_params = {**LGBM_BASE_PARAMS,
               "learning_rate": 0.02, "feature_fraction": 0.07,
               "max_bin": 127}

print(f"=== Experiment {EXP}: leaves/mcs 再チューニング (p=0.27) ===")
print(f"P1ベース(leaves=63, mcs=10, LOSO=15.4725)\n")
print(f"{'leaves':>8}  {'mcs':>5}  {'LOSO':>8}  {'vs P1':>7}")
print("-" * 36)

best_rmse = np.inf; best_cfg = None; best_iters = None

for leaves in [31, 47, 63]:
    for mcs in [5, 10, 20, 30]:
        params = {**base_params, "num_leaves": leaves, "min_child_samples": mcs}

        oof_trans = np.zeros(len(y_train)); iters = []
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
        diff = rmse - 15.4725
        p1_tag = " [P1]" if leaves == 63 and mcs == 10 else ""
        flag   = " <-- best" if rmse < best_rmse else ""
        print(f"  lv={leaves:3d}  mcs={mcs:2d}  {rmse:8.4f}  {diff:+7.4f}{p1_tag}{flag}")

        if rmse < best_rmse:
            best_rmse = rmse; best_cfg = (leaves, mcs); best_iters = iters

lv_b, mcs_b = best_cfg
print(f"\nBest: leaves={lv_b}, mcs={mcs_b}  LOSO={best_rmse:.4f}  vs P1: {best_rmse - 15.4725:+.4f}")

if best_rmse < 15.4725:
    final_params = {**base_params, "num_leaves": lv_b, "min_child_samples": mcs_b}
    dtrain_f = lgb.Dataset(Xtr, label=y_train ** POWER)
    final = lgb.train(final_params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte))
    OUT = f"{OUT_DIR}/submission_{EXP}_lv{lv_b}_mcs{mcs_b}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: leaves={lv_b},mcs={mcs_b} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\n[Skip] P1を超えなかった → 提出なし")
