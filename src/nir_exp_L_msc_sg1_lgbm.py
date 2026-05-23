"""
Experiment L: MSC+SG1 -> LGBM  (Exp Jの正しいバージョン)
Exp J の失敗原因: MSCのみでSG微分なし + num_leaves=10 (設定ミス)
今回: MSC+SG1 vs SNV+SG1 を正しく比較する。
またH実験で得た最良パラメータ(lr=0.02,leaves=63,ff=0.07,mcs=30)でも検証。
Baseline (G): SNV+SG1+LGBM  LOSO=21.48
Best so far (H): LOSO=20.74
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")

from nir_loso_utils import (
    load_data, snv, sg_deriv, msc,
    loso_lgbm,
    save_submission, plot_residuals,
    LGBM_BASE_PARAMS,
)
import warnings
warnings.filterwarnings("ignore")

EXP_LETTER = "L"
OUT_PATH   = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_L_msc_sg1.csv"

data = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

print("=== Experiment L: MSC+SG1 vs SNV+SG1 ===")
print("Baseline (G): LOSO=21.48  Best(H): LOSO=20.74\n")

# ── 参照スペクトル (訓練全体平均) ─────────────────────────────────────────────
ref = X_train_raw.mean(axis=0)

# ── 前処理バリアント ──────────────────────────────────────────────────────────
preproc = {
    "SNV+SG1 (baseline)":  (sg_deriv(snv(X_train_raw)),            sg_deriv(snv(X_test_raw))),
    "MSC+SG1":             (sg_deriv(msc(X_train_raw, ref)),        sg_deriv(msc(X_test_raw, ref))),
    "MSC+SNV+SG1":         (sg_deriv(snv(msc(X_train_raw, ref))),  sg_deriv(snv(msc(X_test_raw, ref)))),
    "SNV+MSC+SG1":         (sg_deriv(msc(snv(X_train_raw))),       sg_deriv(msc(snv(X_test_raw)))),
}

# ── 2種のLGBMパラメータで比較 ────────────────────────────────────────────────
param_sets = {
    "G-params (lr=0.05,leaves=31,ff=0.10,mcs=10)": {**LGBM_BASE_PARAMS},
    "H-params (lr=0.02,leaves=63,ff=0.07,mcs=30)": {
        **LGBM_BASE_PARAMS,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'feature_fraction': 0.07,
        'min_child_samples': 30,
    },
}

best_rmse   = np.inf
best_label  = None
best_oof    = None
best_Xtr    = None
best_Xte    = None
best_params = None

for pp_label, (Xtr, Xte) in preproc.items():
    for p_label, params in param_sets.items():
        rmse, avg_rounds, oof = loso_lgbm(Xtr, y_train, sp_train, params,
                                           n_rounds=2000)
        flag = " <-- best" if rmse < best_rmse else ""
        print(f"  [{pp_label}] + [{p_label}]")
        print(f"    LOSO-RMSE={rmse:.4f}  avg_rounds={avg_rounds}{flag}")
        if rmse < best_rmse:
            best_rmse   = rmse
            best_label  = f"{pp_label} + {p_label}"
            best_oof    = oof.copy()
            best_Xtr    = Xtr
            best_Xte    = Xte
            best_params = params

print(f"\n=== RESULT ===")
print(f"Best: {best_label}")
print(f"LOSO-RMSE = {best_rmse:.4f}")
print(f"Baseline  : 21.48")
print(f"Best (H)  : 20.74")
print(f"Delta vs G: {best_rmse - 21.48:+.4f}")
print(f"Delta vs H: {best_rmse - 20.74:+.4f}")

# ── プロット ─────────────────────────────────────────────────────────────────
plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp L  LOSO={best_rmse:.4f}")

# ── 最終モデル ────────────────────────────────────────────────────────────────
import lightgbm as lgb
dtrain = lgb.Dataset(best_Xtr, label=y_train)
final  = lgb.train(best_params, dtrain,
                   num_boost_round=400,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = final.predict(best_Xte)
save_submission(test_ids, preds, OUT_PATH)
