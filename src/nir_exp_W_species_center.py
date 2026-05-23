"""
Experiment W: 樹種単位スペクトルセンタリング
=================================================
仮説:
  MSC+SG後も樹種ごとに「平均スペクトル形状」が異なる。
  LOSO設定では未知樹種への汎化が鍵であるため、
  各樹種のバッチ内で平均スペクトルを引いてセンタリングすることで
  樹種固有のオフセットを除去できる可能性がある。

実装:
  [Train fold]  各訓練樹種ごとに平均スペクトルを引く
  [Val fold]    ホールドアウト樹種は自身の平均スペクトルを引く
  [Test]        テスト樹種ごとに自身の平均スペクトルを引く
  ※ ラベルを使わない正規化 → リーケージなし

ベース: T (MSC+SG w=5,poly=3 + sqrt(y) + LGBM T-params) LOSO=19.55
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, plot_residuals, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "W"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_W_species_center.csv"

# ── データ読み込み ────────────────────────────────────────────────────────────
data = load_data()
y_train    = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

# テスト樹種ラベルを追加取得
test_df = pd.read_csv(
    r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\test (2).csv",
    encoding="shift-jis"
)
sp_test = test_df["species number"].values

# ── 前処理: MSC+SG(w=5, poly=3) ──────────────────────────────────────────────
ref = X_train_raw.mean(axis=0)
X_tr_pp = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
X_te_pp = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)

y_sqrt = np.sqrt(y_train)

# ── 樹種単位センタリング関数 ───────────────────────────────────────────────────
def species_center_train(X: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """各訓練樹種ごとに平均スペクトルを引く。"""
    X_out = X.copy()
    for s in np.unique(sp):
        mask = sp == s
        X_out[mask] -= X[mask].mean(axis=0)
    return X_out


def species_center_batch(X: np.ndarray) -> np.ndarray:
    """1樹種のバッチ全体を自身の平均でセンタリング。"""
    return X - X.mean(axis=0)


# ── LGBM パラメータ (T-best) ──────────────────────────────────────────────────
params = {
    **LGBM_BASE_PARAMS,
    "learning_rate":    0.02,
    "num_leaves":       63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
}

print("=== Experiment W: Species-level Spectrum Centering ===")
print("Base: T  MSC+SG(w=5,poly=3)+sqrt  LOSO=19.55")
print()

# ── LOSO-CV ──────────────────────────────────────────────────────────────────
oof = np.zeros(len(y_train))

for tr_idx, va_idx, sp in loso_folds(sp_train):
    # センタリング: 訓練樹種は各種の平均を引き、バリデーション種は自身平均を引く
    X_tr_c = species_center_train(X_tr_pp[tr_idx], sp_train[tr_idx])
    X_va_c = species_center_batch(X_tr_pp[va_idx])

    dtrain = lgb.Dataset(X_tr_c, label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(X_va_c, label=y_sqrt[va_idx], reference=dtrain)
    model  = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof[va_idx] = np.clip(model.predict(X_va_c), 0, None) ** 2

rmse_w = loso_rmse(oof, y_train)
print(f"[W] Species-centered  LOSO-RMSE = {rmse_w:.4f}")
print(f"    Delta vs T(19.55): {rmse_w - 19.55:+.4f}")
print()

# 比較: センタリングなし（同パラメータ）
print("--- Comparison: no centering (T-params) ---")
oof_base = np.zeros(len(y_train))
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(X_tr_pp[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(X_tr_pp[va_idx],  label=y_sqrt[va_idx], reference=dtrain)
    model  = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_base[va_idx] = np.clip(model.predict(X_tr_pp[va_idx]), 0, None) ** 2

rmse_base = loso_rmse(oof_base, y_train)
print(f"[Base] No centering  LOSO-RMSE = {rmse_base:.4f}")
print()

# ── 結果サマリ ────────────────────────────────────────────────────────────────
print("=== RESULT SUMMARY ===")
print(f"  No centering : LOSO = {rmse_base:.4f}")
print(f"  Centering    : LOSO = {rmse_w:.4f}  (Delta={rmse_w - rmse_base:+.4f})")

if rmse_w < rmse_base:
    print("  → Centering IMPROVES performance. Use this for submission.")
    best_use_centering = True
else:
    print("  → Centering does NOT help. Stick with no-centering.")
    best_use_centering = False

# ── 残差プロット ──────────────────────────────────────────────────────────────
plot_residuals(
    oof if best_use_centering else oof_base,
    y_train, sp_train, EXP_LETTER,
    title=f"Exp W [species centering={'ON' if best_use_centering else 'OFF'}]  LOSO={min(rmse_w, rmse_base):.4f}",
)

# ── 最終モデル訓練・提出ファイル生成 ──────────────────────────────────────────
if best_use_centering:
    X_tr_final = species_center_train(X_tr_pp, sp_train)
    # テスト: 樹種ごとにセンタリング
    X_te_final = X_te_pp.copy()
    for s in np.unique(sp_test):
        mask = sp_test == s
        X_te_final[mask] = species_center_batch(X_te_pp[mask])
else:
    X_tr_final = X_tr_pp
    X_te_final = X_te_pp

dtrain_final = lgb.Dataset(X_tr_final, label=y_sqrt)
final_model  = lgb.train(
    params, dtrain_final,
    num_boost_round=500,
    callbacks=[lgb.log_evaluation(-1)],
)
preds = np.clip(final_model.predict(X_te_final), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
