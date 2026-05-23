"""
Experiment R: P+Q組み合わせ (window=9 × leaves=31 × mcs=10)
P: window=9 → LOSO=20.10
Q: leaves=31, mcs=10 → LOSO=20.17
組み合わせでさらに改善できるか？
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, plot_residuals, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "R"
OUT_PATH = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_R_combined.csv"

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]
sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
y_sqrt = np.sqrt(y_train)

print("=== Experiment R: P+Q Combined (window=9, leaves=31, mcs=10) ===")
print("P: window=9  LOSO=20.10")
print("Q: leaves=31, mcs=10  LOSO=20.17")
print("R target: <20.10\n")

# window=9 (P best)
X_tr = sg_deriv(msc(X_train_raw, ref), window=9)
X_te = sg_deriv(msc(X_test_raw,  ref), window=9)

# sqrt変換後の最適パラメータでグリッド探索（window=9固定）
base = dict(objective='regression', metric='rmse', verbosity=-1,
            n_jobs=-1, random_state=42, learning_rate=0.02, feature_fraction=0.07)

best_rmse = np.inf; best_cfg = None; best_oof = None

print("--- leaves x mcs grid (window=9, ff=0.07, lr=0.02) ---")
for leaves in [15, 31, 63]:
    for mcs in [5, 10, 20, 30]:
        params = {**base, 'num_leaves': leaves, 'min_child_samples': mcs}
        oof = np.zeros(len(y_train))
        for tr_idx, va_idx, sp in loso_folds(sp_train):
            dtrain = lgb.Dataset(X_tr[tr_idx], label=y_sqrt[tr_idx])
            dval   = lgb.Dataset(X_tr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
            model  = lgb.train(params, dtrain, num_boost_round=2000,
                               valid_sets=[dval],
                               callbacks=[lgb.early_stopping(50, verbose=False),
                                          lgb.log_evaluation(-1)])
            oof[va_idx] = np.clip(model.predict(X_tr[va_idx]), 0, None) ** 2
        rmse = loso_rmse(oof, y_train)
        flag = " <-- best" if rmse < best_rmse else ""
        print(f"  leaves={leaves:3d} mcs={mcs:2d}  LOSO={rmse:.4f}{flag}")
        if rmse < best_rmse:
            best_rmse = rmse; best_cfg = params.copy(); best_oof = oof.copy()

print(f"\n=== RESULT ===")
print(f"Best config: {best_cfg}")
print(f"LOSO-RMSE = {best_rmse:.4f}")
print(f"Delta vs P(20.10): {best_rmse - 20.0994:+.4f}")
print(f"Delta vs G(21.48): {best_rmse - 21.48:+.4f}")

plot_residuals(best_oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp R [w=9+best params]  LOSO={best_rmse:.4f}")

dtrain = lgb.Dataset(X_tr, label=y_sqrt)
final  = lgb.train(best_cfg, dtrain, num_boost_round=500,
                   callbacks=[lgb.log_evaluation(-1)])
preds  = np.clip(final.predict(X_te), 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
