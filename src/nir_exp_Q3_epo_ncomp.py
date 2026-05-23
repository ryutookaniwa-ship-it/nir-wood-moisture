"""
Experiment Q3: EPO n_components 再探索 (p=0.27)
================================================
J1実験: p=0.50(sqrt) で n=5 が最適、n=6以上で急激悪化。
p=0.27 変換後もこの結果が変わらないか確認。
ターゲット分布が変わると EPO の最適な除去成分数も変わる可能性。

探索: n_components = [2, 3, 4, 5, 6, 7, 8]
ベース: P1 (n=5, p=0.27, LOSO=15.4725, LB=15.395)
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

EXP = "Q3"
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

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

y_trans = y_train ** POWER
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / POWER)

print(f"=== Experiment {EXP}: EPO n_components 再探索 (p=0.27) ===")
print(f"P1ベース(n=5, LOSO=15.4725, LB=15.395)\n")
print(f"{'n_comp':>7}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 40)

best_rmse = np.inf; best_n = None; best_iters = None

for n in [2, 3, 4, 5, 6, 7, 8]:
    V = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=n)
    Xtr = apply_epo(Xtr_sg, V)
    Xte = apply_epo(Xte_sg, V)

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
    avg_r = int(np.mean(iters))
    diff = rmse - 15.4725
    p1_tag = " [P1]" if n == 5 else ""
    flag   = " <-- best" if rmse < best_rmse else ""
    print(f"  n={n}  {rmse:8.4f}  {avg_r:9d}  {diff:+7.4f}{p1_tag}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_n = n; best_iters = iters
        best_Xtr = Xtr.copy(); best_Xte = Xte.copy()

print(f"\nBest: n_comp={best_n}  LOSO={best_rmse:.4f}  vs P1: {best_rmse - 15.4725:+.4f}")

if best_rmse < 15.4725:
    dtrain_f = lgb.Dataset(best_Xtr, label=y_train ** POWER)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=int(np.mean(best_iters)),
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(best_Xte))
    OUT = f"{OUT_DIR}/submission_{EXP}_n{best_n}.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{EXP}: EPO n_comp={best_n} LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print("\n[Skip] P1を超えなかった → 提出なし")
