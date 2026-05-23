"""Re-optimize power transform with best LGBM params (ps=10, lv=47).
POWER=0.27 was tuned with P1 params. The optimal may shift with new params.
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")

INPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\4_ds\competitions\signate\nir_wood_moisture\input")

train = pd.read_csv(INPUT_DIR / "train.csv", encoding="shift-jis")
test  = pd.read_csv(INPUT_DIR / "test.csv",  encoding="shift-jis")

target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()

y_train     = train[target_col].values.astype(np.float64)
X_train_raw = train[spec_cols].values.astype(np.float64)
sp_train    = train["species number"].values

def msc(X, reference=None):
    ref = reference if reference is not None else X.mean(axis=0)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        coef = np.polyfit(ref, X[i], 1)
        out[i] = (X[i] - coef[1]) / coef[0]
    return out, ref

def sg_deriv(X, window=9, polyorder=2, deriv=1):
    return savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1)

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
    if not all_dirs:
        return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T

def apply_epo(X, V):
    return X - (X @ V) @ V.T

def loso_folds(sp):
    for s in sorted(set(sp)):
        va = np.where(sp == s)[0]
        tr = np.where(sp != s)[0]
        yield tr, va, s

X_tr_msc, msc_ref = msc(X_train_raw)
X_tr_sg = sg_deriv(X_tr_msc, window=9, polyorder=2, deriv=1)
V_epo = compute_epo_matrix(X_tr_sg, y_train, sp_train, bin_width=10.0, n_components=5)
X_tr_epo = apply_epo(X_tr_sg, V_epo)

for pw in [0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32, 0.35, 0.40, 0.50]:
    y_transformed = np.power(y_train, pw)

    LGBM_PARAMS = dict(
        objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
        random_state=42, learning_rate=0.02, num_leaves=33,
        feature_fraction=0.07, min_child_samples=10,
        path_smooth=7,
    )

    oof = np.zeros(len(y_transformed))
    for tr_idx, va_idx, sp_id in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr_epo[tr_idx], label=y_transformed[tr_idx])
        dval   = lgb.Dataset(X_tr_epo[va_idx], label=y_transformed[va_idx], reference=dtrain)
        model  = lgb.train(
            LGBM_PARAMS, dtrain, num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va_idx] = model.predict(X_tr_epo[va_idx])
    oof_original = np.power(np.clip(oof, 0, None), 1.0 / pw)
    rmse = float(np.sqrt(np.mean((y_train - oof_original) ** 2)))
    tag = " ***" if rmse < 15.2356 else ""
    print(f"POWER={pw:.2f}  RMSE={rmse:.4f}{tag}")
