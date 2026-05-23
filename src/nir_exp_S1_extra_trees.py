"""
Experiment S1: extra_trees=True — ランダム分割で未知樹種汎化向上
================================================================
仮説:
  LightGBMのextra_treesは各特徴量でランダムな閾値を1つだけ評価する。
  通常GBDTより決定境界が滑らかになり、未知ドメインへの汎化が向上する。
  既にff=0.07で特徴量ランダム性が高い → 分割閾値もランダム化で二重抑制。
  P1のgap=-0.08をさらに負方向に動かす可能性。

ベース: P1 (MSC+SG(w=9,p=2)+EPO(n=5)+y^0.27, LOSO=15.4725, LB=15.395)
期待改善: -0.3〜1.0
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

EXP = "S1"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"


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


def run_loso(Xtr, y_trans, sp, params, n_rounds=3000, patience=50):
    oof = np.zeros(len(y_trans)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(patience, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)
    return oof, int(np.mean(iters))


# ── Data & preprocessing (P1 pipeline) ───────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)
y_p027 = y_train ** 0.27

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}

P1_BASELINE = 15.4725  # LOSO
P1_LB       = 15.395

print(f"=== Experiment {EXP}: extra_trees=True ===")
print(f"ベース: P1 (LOSO={P1_BASELINE}, LB={P1_LB})\n")

# ── Test 1: extra_trees=True, same leaves/mcs as P1 ─────────────────────────
params_et = {**P1_PARAMS, "extra_trees": True}
oof_trans, avg_iter = run_loso(Xtr, y_p027, sp_train, params_et)
oof_et = np.clip(oof_trans, 0, None) ** (1.0 / 0.27)
rmse_et = loso_rmse(oof_et, y_train)
print(f"extra_trees=True (leaves=63, mcs=10): LOSO={rmse_et:.4f}  avg_iter={avg_iter}")
print(f"  vs P1: {rmse_et - P1_BASELINE:+.4f}")

# ── Test 2: extra_trees=True + leaves探索 ────────────────────────────────────
print(f"\nextra_trees=True + num_leaves探索:")
print(f"{'leaves':>8}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 38)
best_rmse = rmse_et; best_leaves = 63; best_iter = avg_iter

for lv in [31, 47, 63, 95, 127]:
    params_lv = {**P1_PARAMS, "extra_trees": True, "num_leaves": lv}
    oof_t, ai = run_loso(Xtr, y_p027, sp_train, params_lv)
    rmse = loso_rmse(np.clip(oof_t, 0, None) ** (1/0.27), y_train)
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  lv={lv:3d}    {rmse:8.4f}  {ai:9d}  {diff:+7.4f}{flag}")
    if rmse < best_rmse:
        best_rmse = rmse; best_leaves = lv; best_iter = ai
        best_oof_trans = oof_t.copy()

print(f"\nBest: extra_trees=True, leaves={best_leaves}  LOSO={best_rmse:.4f}")

# ── Submission if improved ────────────────────────────────────────────────────
if best_rmse < P1_BASELINE:
    y_full = y_train ** 0.27
    params_final = {**P1_PARAMS, "extra_trees": True, "num_leaves": best_leaves}
    dtrain_f = lgb.Dataset(Xtr, label=y_full)
    final = lgb.train(params_final, dtrain_f,
                      num_boost_round=best_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte), 0, None) ** (1/0.27)
    OUT = f"{OUT_DIR}/submission_{EXP}_lv{best_leaves}_et.csv"
    save_submission(test_ids, preds, OUT)
    memo = f"{EXP}: extra_trees lv={best_leaves} LOSO={best_rmse:.4f}"
    submit_to_signate(OUT, memo, loso=best_rmse)
else:
    print(f"\n[Skip] P1(LOSO={P1_BASELINE})を超えなかった → 提出なし")
