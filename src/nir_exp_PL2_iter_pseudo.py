"""
Experiment PL2: 反復疑似ラベル (Iterative Pseudo-labeling)
===========================================================
PL1: P1モデルで疑似ラベル生成 → LOSO-0.41 / LB-0.002
PL2: PL1モデルで再予測 → 疑似ラベルを更新 → 再学習

仮説: P1の予測より PL1(テスト種を学習済み)の予測の方が精度が高い
→ より正確な疑似ラベル → さらなる改善

手順:
  Round0: P1 (元訓練データのみ) → テスト疑似ラベルv0
  Round1: P1 + v0 → 再学習 → テスト疑似ラベルv1  (= PL1)
  Round2: P1 + v1 → 再学習 → テスト疑似ラベルv2  (= PL2)
  各ラウンドでLOSO評価(元13種)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

EXP = "PL2"
P_POWER = 0.27
N_ROUNDS_BASE = 600   # P1 avg_iter
N_ROUNDS_PL   = 616   # PL1 avg_iter

P1_PARAMS = {**LGBM_BASE_PARAMS,
             "learning_rate": 0.02, "num_leaves": 63,
             "feature_fraction": 0.07, "min_child_samples": 10}


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
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


def run_loso(X_full, y_aug, y_train, sp_train, sp_aug, params, n_rounds):
    """元の13種のみでLOSO評価。テスト種は常に訓練に含める。"""
    te_indices = np.arange(len(y_train), len(y_aug))
    oof = np.zeros(len(y_train)); best_iters = []
    for tr_idx_orig, va_idx_orig, _ in loso_folds(sp_train):
        tr_idx_aug = np.concatenate([tr_idx_orig, te_indices])
        dtrain = lgb.Dataset(X_full[tr_idx_aug], label=(y_aug[tr_idx_aug] ** P_POWER))
        dval   = lgb.Dataset(X_full[va_idx_orig], label=(y_aug[va_idx_orig] ** P_POWER),
                             reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx_orig] = np.clip(m.predict(X_full[va_idx_orig]), 0, None) ** (1 / P_POWER)
        best_iters.append(m.best_iteration)
    return loso_rmse(oof, y_train), int(np.mean(best_iters)), oof


# ── Load & P1前処理 ─────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
y_train  = train[target_col].values
X_tr_raw = train[spec_cols].values.astype(np.float64)
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_train = train["species number"].values
sp_test  = test["species number"].values

ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

print(f"=== {EXP}: 反復疑似ラベル (2ラウンド) ===\n")

# ── Round0: P1 → 疑似ラベルv0 ────────────────────────────────────────────────
print("Round0: P1全データ学習 → 疑似ラベルv0...")
dtrain0 = lgb.Dataset(Xtr_epo, label=y_train ** P_POWER)
m0 = lgb.train(P1_PARAMS, dtrain0, num_boost_round=N_ROUNDS_BASE,
               callbacks=[lgb.log_evaluation(-1)])
pseudo_v0 = np.clip(m0.predict(Xte_epo), 0, None) ** (1 / P_POWER)
print(f"  v0: mean={pseudo_v0.mean():.2f}")

# ── Round1: P1 + v0 → 疑似ラベルv1 (= PL1相当) ───────────────────────────────
print("\nRound1 (PL1相当): P1 + v0 → LOSO評価 + 疑似ラベルv1...")
X_aug1 = np.vstack([Xtr_epo, Xte_epo])
y_aug1 = np.concatenate([y_train, pseudo_v0])
sp_aug1 = np.concatenate([sp_train, sp_test])

rmse1, avg_r1, _ = run_loso(X_aug1, y_aug1, y_train, sp_train, sp_aug1, P1_PARAMS, 3000)
print(f"  Round1 LOSO: {rmse1:.4f}  (PL1実績=15.0660)")

# Round1の全データモデルで疑似ラベルv1を生成
dtrain1 = lgb.Dataset(X_aug1, label=y_aug1 ** P_POWER)
m1 = lgb.train(P1_PARAMS, dtrain1, num_boost_round=avg_r1,
               callbacks=[lgb.log_evaluation(-1)])
pseudo_v1 = np.clip(m1.predict(Xte_epo), 0, None) ** (1 / P_POWER)
print(f"  v0→v1変化量: mean_diff={np.abs(pseudo_v1 - pseudo_v0).mean():.3f}")

# ── Round2: P1 + v1 → PL2本体 ────────────────────────────────────────────────
print("\nRound2 (PL2): P1 + v1 → LOSO評価...")
X_aug2 = np.vstack([Xtr_epo, Xte_epo])
y_aug2 = np.concatenate([y_train, pseudo_v1])
sp_aug2 = np.concatenate([sp_train, sp_test])

rmse2, avg_r2, _ = run_loso(X_aug2, y_aug2, y_train, sp_train, sp_aug2, P1_PARAMS, 3000)
print(f"  Round2 LOSO: {rmse2:.4f}  (Round1={rmse1:.4f}, delta={rmse2-rmse1:+.4f})")

print(f"\n=== サマリ ===")
print(f"P1    LOSO: 15.4725")
print(f"PL1   LOSO: 15.0660  (LB=15.393)")
print(f"PL2   LOSO: {rmse2:.4f}  delta vs PL1={rmse2-15.0660:+.4f}")

# ── 最終モデル & 提出 ─────────────────────────────────────────────────────────
dtrain_f = lgb.Dataset(X_aug2, label=y_aug2 ** P_POWER)
final = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r2,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: iter_pseudo_v1, LOSO={rmse2:.4f}", loso=rmse2)
print(f"[Done] {EXP}")
