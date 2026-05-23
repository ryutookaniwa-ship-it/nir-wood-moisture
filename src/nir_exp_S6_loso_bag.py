"""
Experiment S6: LOSO Bagging (13樹種×保留モデル平均)
====================================================
仮説:
  テストには「未知樹種」が来る → 各モデルが1樹種を未経験で学習したアンサンブル
  がテスト樹種への汎化に最も近い状況を模倣する。

  通常のシードアンサンブル(I4)は r=0.998 で多様性なし。
  一方、LOSO-baggingは各モデルが異なる「欠損樹種」で学習するため
  意味のある多様性が生まれる。

実装:
  for sp in [1,3,4,5,8,11,12,13,14,15,16,17,19]:  # 13種
      m_sp = LGBM(P1-params, avg_iter_sp)
      m_sp.fit(Xtr[sp_train != sp], y_p027[sp_train != sp])
      preds_sp = m_sp.predict(Xte)

  final_pred = mean(preds_sp) ** (1/0.27)

LOSOスコアの代替計算:
  LOSO fold i: m_i はすでに樹種 s_i を除いて学習済み
  → oof[sp==s_i] = m_i.predict(Xtr[sp==s_i])
  (各モデルの平均予測ではなくood内のモデルのみを使用)

  LOSO上のアンサンブル:
  oof_agg = mean of oof from models that DID NOT see species s_i
  → oof_agg[sp==s_i] = mean over {m_j : j != i} of predictions

ベース: P1 (LOSO=15.4725, LB=15.395)
期待改善: 多様性があればI4(-0.09)より大きく改善
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

EXP = "S6"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

P1_BASELINE = 15.4725
P1_LB       = 15.395

# ── Data & P1 pipeline ────────────────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
y_p027 = y_train ** 0.27


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


V = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_sg, V)
Xte = apply_epo(Xte_sg, V)

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}

# ── Step 1: P1 avg_iter per species held out ──────────────────────────────────
# まず各fold avg_iterを取得（アンサンブルの最終学習に使用）
print(f"=== Experiment {EXP}: LOSO Bagging ===")
print(f"Base: P1 (LOSO={P1_BASELINE}, LB={P1_LB})\n")
print("Step 1: Learning avg_iter per species holdout...", flush=True)

species_list = sorted(set(sp_train))
fold_models = {}  # sp -> (model, avg_iter, tr_idx)

for sp_held in species_list:
    tr_idx = np.where(sp_train != sp_held)[0]
    va_idx = np.where(sp_train == sp_held)[0]
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_p027[va_idx], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
    fold_models[sp_held] = {
        "model": m,
        "best_iter": m.best_iteration,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
    }
    print(f"  sp{sp_held:2d}: best_iter={m.best_iteration}", flush=True)

# ── Step 2: LOSO スコア計算 (cross-species OOF) ─────────────────────────────
# Strategy: For each held-out species s_i, average predictions from
# all models that did NOT see s_i = all models except m_i itself
# (m_i never saw s_i, so its prediction is the "true" OOF)
# Simple: just use each species' own held-out model prediction

oof_p1_style = np.zeros(len(y_train))
for sp_held, info in fold_models.items():
    va_idx = info["va_idx"]
    oof_p1_style[va_idx] = info["model"].predict(Xtr[va_idx])

loso_individual = loso_rmse(np.clip(oof_p1_style, 0, None) ** (1/0.27), y_train)
print(f"\nStep 2: Individual model LOSO = {loso_individual:.4f} (should match P1~{P1_BASELINE})")

# ── Step 3: テスト予測アンサンブル (全13モデル平均) ──────────────────────────
print("\nStep 3: Building final ensemble predictions...")

# 全13モデルの最終学習 (avg_iter each)
test_preds_all = []
for sp_held, info in fold_models.items():
    tr_idx = info["tr_idx"]
    n_rounds = info["best_iter"]
    dtrain_f = lgb.Dataset(Xtr[tr_idx], label=y_p027[tr_idx])
    m_final = lgb.train(P1_PARAMS, dtrain_f,
                        num_boost_round=n_rounds,
                        callbacks=[lgb.log_evaluation(-1)])
    pred_te = m_final.predict(Xte)
    test_preds_all.append(pred_te)
    print(f"  sp{sp_held:2d}: n_rounds={n_rounds}", flush=True)

test_preds_mean = np.mean(test_preds_all, axis=0)
preds_final = np.clip(test_preds_mean, 0, None) ** (1/0.27)

# ── Step 4: アンサンブル間の相関確認 ─────────────────────────────────────────
print("\nStep 4: Inter-model correlation analysis")
preds_matrix = np.array(test_preds_all)
corr_matrix = np.corrcoef(preds_matrix)
# Upper triangle only
triu = corr_matrix[np.triu_indices(len(species_list), k=1)]
print(f"  Pred correlation: mean={triu.mean():.4f}, min={triu.min():.4f}, max={triu.max():.4f}")

print(f"\nFinal ensemble LOSO (individual fold estimate) = {loso_individual:.4f}")
print(f"(Note: true ensemble LOSO would require re-averaging OOF, using individual as proxy)")

# ── Submission ────────────────────────────────────────────────────────────────
OUT = f"{OUT_DIR}/submission_{EXP}_loso_bag13.csv"
save_submission(test_ids, preds_final, OUT)
memo = f"{EXP}: LOSO-Bag(13models) corr_mean={triu.mean():.3f}"
# Use P1_BASELINE as LOSO proxy (true LOSO of ensemble unknown until submission)
# Submit regardless of threshold since this is a qualitatively different approach
submit_to_signate(OUT, memo, loso=loso_individual)
