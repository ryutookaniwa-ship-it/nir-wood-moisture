"""
Experiment I4: B2 シードアンサンブル
=================================================
同一B2設定を異なるrandom_seedで複数学習して平均。
LGBMのブートストラップ/特徴サンプリングのランダム性を活用。
モデルアーキテクチャは同一だが予測のばらつきを平均で減らす。

手順:
  seeds: [42, 7, 13, 21, 99]  5モデル
  各seedで完全なLOSO-CVを回してOOF平均でLOSO評価
  最終予測 = 5モデルのテスト予測平均

ベース: B2 (LOSO=16.44, LB=17.651)
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

EXP   = "I4"
SEEDS = [42, 7, 13, 21, 99]

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
Xtr_pp = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
Xte_pp = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)
V = compute_epo_matrix(Xtr_pp, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_pp, V); Xte_epo = apply_epo(Xte_pp, V)
y_sqrt = np.sqrt(y_train)

base_params = {**LGBM_BASE_PARAMS, "learning_rate": 0.02, "num_leaves": 63,
               "feature_fraction": 0.07, "min_child_samples": 10}

print(f"=== Experiment {EXP}: B2 シードアンサンブル ===")
print(f"Seeds: {SEEDS}\n")

all_oof   = []
all_preds = []

for seed in SEEDS:
    params = {**base_params, "random_state": seed, "feature_fraction_seed": seed}
    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx, va_idx, sp in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(Xtr_epo[va_idx]), 0, None) ** 2
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_train)
    print(f"  seed={seed:3d}  LOSO={rmse:.4f}  avg_iter={int(np.mean(best_iters))}")
    all_oof.append(oof)

    # 最終モデル（全訓練データ）
    avg_rounds = int(np.mean(best_iters))
    dtrain_f = lgb.Dataset(Xtr_epo, label=y_sqrt)
    final = lgb.train(params, dtrain_f, num_boost_round=avg_rounds,
                      callbacks=[lgb.log_evaluation(-1)])
    all_preds.append(np.clip(final.predict(Xte_epo), 0, None) ** 2)

# アンサンブル評価
oof_ensemble = np.mean(all_oof, axis=0)
rmse_ens = loso_rmse(oof_ensemble, y_train)

# 個別seedとの相関
print(f"\nシード間OOF相関:")
for i in range(len(SEEDS)):
    for j in range(i+1, len(SEEDS)):
        r = np.corrcoef(all_oof[i], all_oof[j])[0, 1]
        print(f"  seed{SEEDS[i]} vs seed{SEEDS[j]}: r={r:.4f}")

print(f"\nアンサンブル LOSO={rmse_ens:.4f}")
print(f"vs B2(16.44): {rmse_ens - 16.44:+.4f}")

preds_final = np.mean(all_preds, axis=0)
OUT = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\submission_I4_seed_ensemble.csv"
save_submission(test_ids, preds_final, OUT)
print(f"\n[Done] {OUT}")
