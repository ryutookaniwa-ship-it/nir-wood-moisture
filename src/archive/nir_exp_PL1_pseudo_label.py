"""
Experiment PL1: 疑似ラベル (Pseudo-labeling)
============================================
P1でテストデータを予測し、疑似ラベルとして訓練に追加して再学習。
「未知樹種6種のスペクトルを学習に含める」ことで domain shift を縮小。

手順:
  1. P1を全訓練データで学習し、テスト550サンプルの疑似ラベルを生成
  2. 訓練データにテストデータ(疑似ラベル付き)を追加
  3. 拡張データセットで P1 を再学習
  4. LOSO評価: 元の13訓練種のみでLOSO (疑似ラベル種は除外)
     → P1比でLOSO劣化がないかチェック
  5. 提出 (真の評価はLBスコア)

注意: EPOは元の訓練データで計算済みのものを流用。
      MSCの参照も元の訓練平均のまま。
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

EXP = "PL1"
P_POWER = 0.27

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


train = pd.read_csv(TRAIN_PATH, encoding="shift-jis")
test  = pd.read_csv(TEST_PATH,  encoding="shift-jis")
target_col = train.columns[3]; spec_cols = train.columns[4:].tolist()
y_train  = train[target_col].values
X_tr_raw = train[spec_cols].values.astype(np.float64)
X_te_raw = test[spec_cols].values.astype(np.float64)
test_ids = test["sample number"].values
sp_train = train["species number"].values
sp_test  = test["species number"].values

# ── P1前処理 ──────────────────────────────────────────────────────────────────
ref = X_tr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_tr_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_te_raw, ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

y_pow = y_train ** P_POWER

# ── Step1: P1でテスト疑似ラベル生成 ───────────────────────────────────────────
print("Step1: P1全データ学習 → テスト疑似ラベル生成...")
# P1のavg_iter≈600を使用
dtrain_orig = lgb.Dataset(Xtr_epo, label=y_pow)
model_p1 = lgb.train(P1_PARAMS, dtrain_orig, num_boost_round=600,
                     callbacks=[lgb.log_evaluation(-1)])
pseudo_labels = np.clip(model_p1.predict(Xte_epo), 0, None) ** (1 / P_POWER)
print(f"  疑似ラベル: min={pseudo_labels.min():.1f}, max={pseudo_labels.max():.1f}, "
      f"mean={pseudo_labels.mean():.1f}")

# ── Step2: 拡張データセット作成 ───────────────────────────────────────────────
X_aug = np.vstack([Xtr_epo, Xte_epo])           # (1322+550, 1555)
y_aug = np.concatenate([y_train, pseudo_labels]) # (1872,)
y_aug_pow = y_aug ** P_POWER
sp_aug = np.concatenate([sp_train, sp_test])     # (1872,)

print(f"  拡張後サンプル数: {len(y_aug)} (train:{len(y_train)} + test:{len(pseudo_labels)})")
print(f"  拡張後樹種数: {len(np.unique(sp_aug))} (訓練13 + テスト6)")

# ── Step3: 拡張データで再学習し、元の13種のみでLOSO評価 ───────────────────────
print("\nStep3: 拡張データでP1再学習 → LOSO(元13種)評価...")
oof = np.zeros(len(y_train))  # 元の訓練データのみ評価
best_iters = []

for tr_idx_orig, va_idx_orig, sp in loso_folds(sp_train):
    # 訓練: 元の12種 + テスト6種(疑似ラベル)
    # バリデーション: 元の1種 (疑似ラベルなし)
    te_indices = np.arange(len(y_train), len(y_aug))  # テストサンプルのインデックス
    tr_idx_aug = np.concatenate([tr_idx_orig, te_indices])

    dtrain = lgb.Dataset(X_aug[tr_idx_aug], label=y_aug_pow[tr_idx_aug])
    dval   = lgb.Dataset(X_aug[va_idx_orig], label=y_aug_pow[va_idx_orig], reference=dtrain)
    m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx_orig] = np.clip(m.predict(X_aug[va_idx_orig]), 0, None) ** (1 / P_POWER)
    best_iters.append(m.best_iteration)

rmse = loso_rmse(oof, y_train)
avg_r = int(np.mean(best_iters))
print(f"\nPL1 LOSO-RMSE (元13種): {rmse:.4f}")
print(f"P1  LOSO-RMSE          : 15.4725")
print(f"Delta                  : {rmse - 15.4725:+.4f}")
print(f"avg_iter               : {avg_r}")

# ── 最終モデル (拡張全データ) ──────────────────────────────────────────────────
dtrain_f = lgb.Dataset(X_aug, label=y_aug_pow)
final = lgb.train(P1_PARAMS, dtrain_f, num_boost_round=avg_r,
                  callbacks=[lgb.log_evaluation(-1)])
preds = np.clip(final.predict(Xte_epo), 0, None) ** (1 / P_POWER)

OUT = str(BASE_DIR / "output" / "nir-wood-moisture" / f"submission_{EXP}.csv")
save_submission(test_ids, preds, OUT)
submit_to_signate(OUT, memo=f"{EXP}: pseudo_label, LOSO={rmse:.4f}", loso=rmse)
print(f"[Done] {EXP}")
