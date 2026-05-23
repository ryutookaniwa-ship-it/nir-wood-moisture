"""
NIR Wood Moisture Content Prediction
Competition: Near-infrared spectra -> wood moisture content (%)
Metric: RMSE
Approach: PLS latent variables + Ridge regression, with SNV preprocessing
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/train (1).csv"
TEST_PATH  = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/test (2).csv"
OUT_PATH   = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/output/submission_nir_pls.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH, encoding='shift-jis')
test  = pd.read_csv(TEST_PATH,  encoding='shift-jis')

target_col    = train.columns[3]             # moisture content
spec_cols     = train.columns[4:].tolist()   # 1555 spectral wavenumbers

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Spectral features: {len(spec_cols)}, range {spec_cols[0]} ~ {spec_cols[-1]} cm-1")

y_train       = train[target_col].values
X_train_raw   = train[spec_cols].values.astype(np.float64)
X_test_raw    = test[spec_cols].values.astype(np.float64)
test_ids      = test['sample number'].values

# ── Preprocessing ──────────────────────────────────────────────────────────────
def snv(X):
    """Standard Normal Variate: normalize each sample spectrum."""
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    return (X - mean) / np.where(std == 0, 1, std)

def sg_deriv(X, window=11, polyorder=2, deriv=1):
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=deriv, axis=1)

# ── CV helper ──────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model, X, y):
    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',
                             cv=kf, n_jobs=-1)
    return -scores.mean(), scores.std()

# ── Step 1: Find best PLS n_components with SNV ────────────────────────────────
print("\n=== PLS n_components search (SNV preprocessing) ===")
X_snv_tr = snv(X_train_raw)
X_snv_te = snv(X_test_raw)

best_pls_rmse = np.inf
best_n_comp   = None
for n in [10, 15, 20, 25, 30, 35, 40, 50]:
    pls  = PLSRegression(n_components=n, max_iter=500)
    rmse, std = cv_rmse(pls, X_snv_tr, y_train)
    flag = " <-- best" if rmse < best_pls_rmse else ""
    print(f"  n_comp={n:2d}  CV-RMSE={rmse:.4f} +/- {std:.4f}{flag}")
    if rmse < best_pls_rmse:
        best_pls_rmse = rmse
        best_n_comp   = n

print(f"\nBest PLS n_comp: {best_n_comp}, CV-RMSE={best_pls_rmse:.4f}")

# ── Step 2: Ridge on PLS latent variables ──────────────────────────────────────
print(f"\n=== Ridge regression on PLS({best_n_comp}) latent variables ===")
pls_final = PLSRegression(n_components=best_n_comp, max_iter=500)
pls_final.fit(X_snv_tr, y_train)
X_lv_tr = pls_final.transform(X_snv_tr)
X_lv_te = pls_final.transform(X_snv_te)

best_ridge_rmse = np.inf
best_alpha      = None
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    rmse, std = cv_rmse(ridge, X_lv_tr, y_train)
    flag = " <-- best" if rmse < best_ridge_rmse else ""
    print(f"  alpha={alpha:7.3f}  CV-RMSE={rmse:.4f} +/- {std:.4f}{flag}")
    if rmse < best_ridge_rmse:
        best_ridge_rmse = rmse
        best_alpha      = alpha

print(f"\nBest Ridge alpha: {best_alpha}, CV-RMSE={best_ridge_rmse:.4f}")

# ── Step 3: Retrain on all data & predict ──────────────────────────────────────
print("\n=== Training final model on all train data ===")
ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(X_lv_tr, y_train)
preds = ridge_final.predict(X_lv_te).ravel()
preds = np.clip(preds, 0, None)

print(f"Predictions: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

# ── Step 4: Also try SNV+SG1 approach ─────────────────────────────────────────
print("\n=== Also testing SNV+SG1 preprocessing ===")
X_sg1_tr = sg_deriv(snv(X_train_raw))
X_sg1_te = sg_deriv(snv(X_test_raw))

best_sg1_rmse = np.inf
best_sg1_comp = None
for n in [10, 15, 20, 25, 30, 35]:
    pls  = PLSRegression(n_components=n, max_iter=500)
    rmse, std = cv_rmse(pls, X_sg1_tr, y_train)
    flag = " <-- best" if rmse < best_sg1_rmse else ""
    print(f"  [snv+sg1] n_comp={n:2d}  CV-RMSE={rmse:.4f} +/- {std:.4f}{flag}")
    if rmse < best_sg1_rmse:
        best_sg1_rmse = rmse
        best_sg1_comp = n

# Ridge on sg1 PLS latents
pls_sg1 = PLSRegression(n_components=best_sg1_comp, max_iter=500)
pls_sg1.fit(X_sg1_tr, y_train)
X_sg1_lv_tr = pls_sg1.transform(X_sg1_tr)
X_sg1_lv_te = pls_sg1.transform(X_sg1_te)

best_sg1_ridge = np.inf
best_sg1_alpha = None
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    rmse, std = cv_rmse(ridge, X_sg1_lv_tr, y_train)
    flag = " <-- best" if rmse < best_sg1_ridge else ""
    print(f"  [snv+sg1 Ridge] alpha={alpha:7.3f}  CV-RMSE={rmse:.4f} +/- {std:.4f}{flag}")
    if rmse < best_sg1_ridge:
        best_sg1_ridge = rmse
        best_sg1_alpha = alpha

print(f"\nSNV CV-RMSE={best_ridge_rmse:.4f} vs SNV+SG1 CV-RMSE={best_sg1_ridge:.4f}")

# ── Choose best approach ───────────────────────────────────────────────────────
if best_ridge_rmse <= best_sg1_ridge:
    print("Using SNV+PLS+Ridge as final model")
    final_preds = preds
else:
    print("Using SNV+SG1+PLS+Ridge as final model")
    ridge_sg1 = Ridge(alpha=best_sg1_alpha)
    ridge_sg1.fit(X_sg1_lv_tr, y_train)
    final_preds = np.clip(ridge_sg1.predict(X_sg1_lv_te).ravel(), 0, None)

# ── Save submission ────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
sub = pd.DataFrame({'id': test_ids, 'pred': final_preds})
sub.to_csv(OUT_PATH, index=False, header=False)
print(f"\nSaved: {OUT_PATH}")
print(sub.head())
print(f"Final CV-RMSE: {min(best_ridge_rmse, best_sg1_ridge):.4f}")
