"""
Experiment S2: Huber損失 + y^0.27変換
======================================
仮説:
  M1実験(Huber探索)はy変換前(I2ベース)で失敗。
  しかし y^0.27変換後はスケールが変わり(max≈3.6)、sp15の外れ値影響が別になる。
  Huber delta=3付近で外れ値をsoft downweightすることでテスト樹種へのバイアスを軽減。

M1失敗の原因: 変換なし → sp15(y=298%)の勾配支配が大きすぎた
今回: y^0.27空間(max≈3.6)で適切なdeltaを探索

ベース: P1 (MSC+SG(w=9,p=2)+EPO(n=5)+y^0.27, LOSO=15.4725, LB=15.395)
期待改善: -0.2〜0.5
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

EXP = "S2"
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

# y^0.27空間での統計確認
print(f"y^0.27空間: min={y_p027.min():.3f}, max={y_p027.max():.3f}, "
      f"std={y_p027.std():.3f}, median={np.median(y_p027):.3f}")

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}

P1_BASELINE = 15.4725
P1_LB       = 15.395

print(f"\n=== Experiment {EXP}: Huber損失 in y^0.27空間 ===")
print(f"ベース: P1 (LOSO={P1_BASELINE}, LB={P1_LB})")
print(f"\n{'delta':>8}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}")
print("-" * 40)

best_rmse = np.inf; best_delta = None; best_iter = None

# deltaグリッド: y^0.27空間でのスケール(max≈3.6)に合わせた探索
for delta in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    params_h = {**P1_PARAMS,
                "objective": "huber",
                "huber_delta": delta,
                "metric": "huber"}

    oof_trans = np.zeros(len(y_p027)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
        m = lgb.train(params_h, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)

    oof  = np.clip(oof_trans, 0, None) ** (1.0 / 0.27)
    rmse = loso_rmse(oof, y_train)
    ai   = int(np.mean(iters))
    diff = rmse - P1_BASELINE
    flag = " <-- best" if rmse < best_rmse else ""
    print(f"  delta={delta:4.1f}   {rmse:8.4f}  {ai:9d}  {diff:+7.4f}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse; best_delta = delta; best_iter = ai

print(f"\nBest: delta={best_delta}  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

# ── Submission if improved ────────────────────────────────────────────────────
if best_rmse < P1_BASELINE:
    params_final = {**P1_PARAMS,
                    "objective": "huber",
                    "huber_delta": best_delta,
                    "metric": "huber"}
    dtrain_f = lgb.Dataset(Xtr, label=y_p027)
    final = lgb.train(params_final, dtrain_f,
                      num_boost_round=best_iter,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(Xte), 0, None) ** (1/0.27)
    OUT = f"{OUT_DIR}/submission_{EXP}_huber_d{int(best_delta*10)}.csv"
    save_submission(test_ids, preds, OUT)
    memo = f"{EXP}: Huber(delta={best_delta})+p=0.27 LOSO={best_rmse:.4f}"
    submit_to_signate(OUT, memo, loso=best_rmse)
else:
    print(f"\n[Skip] P1(LOSO={P1_BASELINE})を超えなかった → 提出なし")
