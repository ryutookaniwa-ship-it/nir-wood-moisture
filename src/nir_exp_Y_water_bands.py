"""
Experiment Y: 水吸収帯精密絞り込み
=======================================
仮説:
  樹種固有パターン（セルロース/リグニン由来の吸収帯）を物理的に除去し、
  水分子に固有のOH吸収帯のみを使うことで未知樹種への汎化が向上する。

主要OH吸収帯（水分子由来）:
  5187 cm⁻¹: OH伸縮+変角 組み合わせ音 (最重要)
  6896 cm⁻¹: OH伸縮 第1倍音          (最重要)
  8333 cm⁻¹: OH伸縮 第2倍音          (中程度)

除外したい樹種固有バンド（妨害帯）:
  ~4760 cm⁻¹: セルロース OH+CH組み合わせ
  ~4400 cm⁻¹: セルロース/ヘミセルロース CH
  ~5900-6000 cm⁻¹: CH伸縮 第1倍音
  ~5600-5800 cm⁻¹: リグニン 芳香族CH

前回 Exp I: [4800-7300] 648点 → LOSO=20.30 (全波長19.55より悪い)
今回: 真に狭いOH帯のみに絞り込む

ベース: T (MSC+SG w=5,poly=3 + sqrt + T-params) LOSO=19.55
"""

import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "Y"
BEST_OUT   = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_Y_water_bands.csv"

# ── データ・基本前処理 ─────────────────────────────────────────────────────────
data        = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]
wns         = data["wns"]

ref    = X_train_raw.mean(axis=0)
X_tr_pp = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3)
X_te_pp = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3)
y_sqrt  = np.sqrt(y_train)

# ── 帯域選択ヘルパー ────────────────────────────────────────────────────────────
def select_bands(X: np.ndarray, wns: np.ndarray, bands: list) -> np.ndarray:
    """bands: [(lo1,hi1), (lo2,hi2), ...] の各帯域を結合して返す。"""
    masks = [((wns >= lo) & (wns <= hi)) for lo, hi in bands]
    mask  = np.logical_or.reduce(masks)
    return X[:, mask], mask.sum()


# ── LGBM パラメータ (T-best ベース) ──────────────────────────────────────────
params_base = {
    **LGBM_BASE_PARAMS,
    "learning_rate":    0.02,
    "num_leaves":       63,
    "min_child_samples": 10,
}

def run_loso(X_tr: np.ndarray, X_te: np.ndarray, ff: float) -> tuple:
    """LOSO-CV + 最終モデル予測。(loso_rmse, preds, oof) を返す。"""
    params = {**params_base, "feature_fraction": ff}
    oof = np.zeros(len(y_train))
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_sqrt[tr_idx])
        dval   = lgb.Dataset(X_tr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=2000,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = np.clip(model.predict(X_tr[va_idx]), 0, None) ** 2
    rmse = loso_rmse(oof, y_train)

    dtrain_f = lgb.Dataset(X_tr, label=y_sqrt)
    final    = lgb.train(params, dtrain_f, num_boost_round=500,
                         callbacks=[lgb.log_evaluation(-1)])
    preds = np.clip(final.predict(X_te), 0, None) ** 2
    return rmse, preds, oof


# ── 帯域候補の定義 ─────────────────────────────────────────────────────────────
# キー: 帯域の組み合わせ説明
# 値: [(lo, hi), ...] のリスト
BAND_CONFIGS = {
    # 2帯域: 5187+6896 (狭め)
    "2B_narrow":  [(5050, 5350), (6700, 7100)],
    # 2帯域: 5187+6896 (広め)
    "2B_wide":    [(5000, 5400), (6600, 7200)],
    # 3帯域: 5187+6896+8333 (狭め)
    "3B_narrow":  [(5050, 5350), (6700, 7100), (8100, 8600)],
    # 3帯域: 5187+6896+8333 (広め)
    "3B_wide":    [(5000, 5400), (6600, 7200), (8100, 8600)],
    # 6896のみ (最重要帯域単独)
    "6896_only":  [(6700, 7100)],
    # 5187+6896+Hbond (6707-7082) — 実質的に6600-7200と同じ
    "2B_Hbond":   [(5050, 5350), (6600, 7200)],
    # 妨害帯を避けた広帯域 (5350-5600を除外してリグニン帯を回避)
    "skip_lign":  [(5050, 5350), (6700, 7100), (7200, 7600), (8100, 8600)],
}

print("=== Experiment Y: 水吸収帯精密絞り込み ===")
print(f"Base T (全1555点):  LOSO=19.55")
print(f"Exp I  (648点帯域): LOSO=20.30")
print()
print(f"{'Config':<14} {'Points':>6}  {'ff':>5}  {'LOSO-RMSE':>10}  {'vs T':>8}")
print("-" * 55)

best_rmse  = np.inf
best_preds = None
best_name  = None
results    = {}

for name, bands in BAND_CONFIGS.items():
    X_tr_sel, n_pts = select_bands(X_tr_pp, wns, bands)
    X_te_sel, _     = select_bands(X_te_pp, wns, bands)

    # 特徴量が少ないほど ff を上げる（各木で最低10特徴を使えるよう）
    ff = min(0.5, max(0.07, 10 / n_pts))

    rmse, preds, _ = run_loso(X_tr_sel, X_te_sel, ff)
    delta = rmse - 19.55
    flag  = " <-- best" if rmse < best_rmse else ""
    print(f"{name:<14} {n_pts:>6}  {ff:>5.2f}  {rmse:>10.4f}  {delta:>+8.4f}{flag}")

    results[name] = (rmse, preds)
    if rmse < best_rmse:
        best_rmse  = rmse
        best_preds = preds
        best_name  = name

print()
print(f"=== RESULT ===")
print(f"Best config : {best_name}")
print(f"Best LOSO   : {best_rmse:.4f}  (vs T: {best_rmse - 19.55:+.4f})")

if best_rmse < 19.55:
    print("→ 帯域絞り込みで改善！提出ファイルを生成します。")
    save_submission(test_ids, best_preds, BEST_OUT)
else:
    print("→ 全波長(T)が依然最良。参考として提出ファイルを生成します。")
    save_submission(test_ids, best_preds, BEST_OUT)
