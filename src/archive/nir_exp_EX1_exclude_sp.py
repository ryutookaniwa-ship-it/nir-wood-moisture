"""
Experiment EX1: 公式リポジトリ準拠 — ベイスギ(sp15)+ベイマツ(sp17)を訓練除外
==========================================================================
公式 dataset.py が EXCLUDED_TRAIN_SPECIES = ["ベイスギ", "ベイマツ"] を
ハードコードしていた。

仮説:
- sp15(ベイスギ)はLOSOの最大ボトルネック(RMSE=54+)
- sp17(ベイマツ)も北米針葉樹として sp15 と類似スペクトル特性を持つ可能性
- これら2種を除外することで EPO の方向推定が安定し、汎化が改善する可能性

試行:
  EX1-A: sp15+sp17を除外、P1パイプライン、LOSO(11種)
  EX1-B: sp15のみ除外(sp17は残す)
  EX1-C: EX1-A + 疑似ラベル(6テスト種)

ベースライン比較:
  P1  : LOSO(13種)=15.4725, LB=15.395
  PL2 : LOSO(13種)=15.0623, LB=15.392
"""
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
    TRAIN_PATH, TEST_PATH, BASE_DIR,
)

EXP = "EX1"
P_POWER = 0.27
EXCLUDE_SP = [15, 17]   # ベイスギ, ベイマツ

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


def train_and_predict(X_tr, y_tr, sp_tr, X_te, params, n_rounds, tag=""):
    oof = np.zeros(len(y_tr)); best_iters = []
    for tr_idx, va_idx, sp_out in loso_folds(sp_tr):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx] ** P_POWER)
        dval   = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx] ** P_POWER, reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(m.predict(X_tr[va_idx]), 0, None) ** (1 / P_POWER)
        best_iters.append(m.best_iteration)
    rmse = loso_rmse(oof, y_tr)
    avg_iter = int(np.mean(best_iters))
    print(f"  {tag} LOSO={rmse:.4f}, avg_iter={avg_iter}")

    # 全データで再学習して予測
    dtrain_full = lgb.Dataset(X_tr, label=y_tr ** P_POWER)
    m_full = lgb.train(params, dtrain_full, num_boost_round=avg_iter,
                       callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(m_full.predict(X_te), 0, None) ** (1 / P_POWER)
    return rmse, avg_iter, oof, preds


# ── データ読み込み ─────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()

y_all_train  = train[target_col].values
X_all_raw    = train[spec_cols].values.astype(np.float64)
sp_all_train = train["species number"].values

X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_test  = test["species number"].values

print(f"全訓練サンプル: {len(y_all_train)} (13種)")
for sp_ex in EXCLUDE_SP:
    n = (sp_all_train == sp_ex).sum()
    print(f"  除外: sp{sp_ex} → {n}サンプル削除")

# ── A: sp15+sp17を除外 ────────────────────────────────────────────────────────
print(f"\n=== EX1-A: sp{EXCLUDE_SP}除外 P1パイプライン ===")
keep_mask = ~np.isin(sp_all_train, EXCLUDE_SP)
X_tr_raw = X_all_raw[keep_mask]
y_tr     = y_all_train[keep_mask]
sp_tr    = sp_all_train[keep_mask]
print(f"使用サンプル数: {len(y_tr)} ({len(np.unique(sp_tr))}種)")

ref_A    = X_tr_raw.mean(axis=0)
Xtr_sg_A = sg_deriv(msc(X_tr_raw, ref_A), window=9, polyorder=2)
Xte_sg_A = sg_deriv(msc(X_te_raw, ref_A), window=9, polyorder=2)
V_A      = compute_epo_matrix(Xtr_sg_A, y_tr, sp_tr, n_components=5)
Xtr_epo_A = apply_epo(Xtr_sg_A, V_A)
Xte_epo_A = apply_epo(Xte_sg_A, V_A)

rmse_A, iter_A, oof_A, preds_A = train_and_predict(
    Xtr_epo_A, y_tr, sp_tr, Xte_epo_A, P1_PARAMS, 3000, tag="EX1-A")

# ── B: sp15のみ除外 ───────────────────────────────────────────────────────────
print(f"\n=== EX1-B: sp15のみ除外 ===")
keep_mask_B = sp_all_train != 15
X_tr_raw_B = X_all_raw[keep_mask_B]
y_tr_B     = y_all_train[keep_mask_B]
sp_tr_B    = sp_all_train[keep_mask_B]
print(f"使用サンプル数: {len(y_tr_B)} ({len(np.unique(sp_tr_B))}種)")

ref_B    = X_tr_raw_B.mean(axis=0)
Xtr_sg_B = sg_deriv(msc(X_tr_raw_B, ref_B), window=9, polyorder=2)
Xte_sg_B = sg_deriv(msc(X_te_raw, ref_B), window=9, polyorder=2)
V_B      = compute_epo_matrix(Xtr_sg_B, y_tr_B, sp_tr_B, n_components=5)
Xtr_epo_B = apply_epo(Xtr_sg_B, V_B)
Xte_epo_B = apply_epo(Xte_sg_B, V_B)

rmse_B, iter_B, oof_B, preds_B = train_and_predict(
    Xtr_epo_B, y_tr_B, sp_tr_B, Xte_epo_B, P1_PARAMS, 3000, tag="EX1-B")

# ── C: EX1-A + 疑似ラベル ─────────────────────────────────────────────────────
print(f"\n=== EX1-C: EX1-A + 疑似ラベル(6テスト種) ===")
X_aug_C  = np.vstack([Xtr_epo_A, Xte_epo_A])
y_aug_C  = np.concatenate([y_tr, preds_A])
sp_aug_C = np.concatenate([sp_tr, sp_test])

# LOSO は元の除外後11種のみで評価
te_idx_C = np.arange(len(y_tr), len(y_aug_C))
oof_C = np.zeros(len(y_tr)); best_iters_C = []
for tr_idx, va_idx, _ in loso_folds(sp_tr):
    tr_idx_aug = np.concatenate([tr_idx, te_idx_C])
    dtrain = lgb.Dataset(X_aug_C[tr_idx_aug], label=y_aug_C[tr_idx_aug] ** P_POWER)
    dval   = lgb.Dataset(X_aug_C[va_idx],     label=y_aug_C[va_idx] ** P_POWER, reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_C[va_idx] = np.clip(m.predict(X_aug_C[va_idx]), 0, None) ** (1 / P_POWER)
    best_iters_C.append(m.best_iteration)

rmse_C = loso_rmse(oof_C, y_tr)
iter_C = int(np.mean(best_iters_C))
print(f"  EX1-C LOSO={rmse_C:.4f}, avg_iter={iter_C}")

dtrain_C = lgb.Dataset(X_aug_C, label=y_aug_C ** P_POWER)
m_C = lgb.train(P1_PARAMS, dtrain_C, num_boost_round=iter_C,
                callbacks=[lgb.log_evaluation(-1)])
preds_C = np.clip(m_C.predict(Xte_epo_A), 0, None) ** (1 / P_POWER)

# ── サマリ ────────────────────────────────────────────────────────────────────
print(f"\n=== {EXP} サマリ ===")
print(f"P1  (13種, LOSO13): 15.4725  (LB=15.395)")
print(f"PL2 (13種, LOSO13): 15.0623  (LB=15.392)")
print(f"EX1-A (11種除外, LOSO11): {rmse_A:.4f}  delta_vs_P1={rmse_A-15.4725:+.4f}")
print(f"EX1-B (12種, sp15のみ除外, LOSO12): {rmse_B:.4f}  delta_vs_P1={rmse_B-15.4725:+.4f}")
print(f"EX1-C (EX1-A+疑似ラベル): {rmse_C:.4f}  delta_vs_A={rmse_C-rmse_A:+.4f}")
print()
print("注意: LOSO種数が異なるため直接比較は参考値。LBでの確認が必須。")

# ── 最良を提出 ────────────────────────────────────────────────────────────────
# LOSOが最も低いものを選んで提出
best = min([("EX1-A", rmse_A, preds_A, iter_A),
            ("EX1-B", rmse_B, preds_B, iter_B),
            ("EX1-C", rmse_C, preds_C, iter_C)],
           key=lambda x: x[1])

best_tag, best_rmse, best_preds, _ = best
print(f"\n提出候補: {best_tag} (LOSO={best_rmse:.4f})")

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, best_preds, OUT)

submit_to_signate(OUT, memo=f"{EXP}: {best_tag}, LOSO={best_rmse:.4f}", loso=best_rmse)

print(f"\n[Done] {EXP}")
