import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")

INPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\input\nir-wood-moisture")
OUTPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\output\nir-wood-moisture")
EXP_ID = "exp_003"

# ── Data loading (Shift-JIS encoded) ──
train = pd.read_csv(INPUT_DIR / "train.csv", encoding="shift-jis")
test  = pd.read_csv(INPUT_DIR / "test.csv",  encoding="shift-jis")

target_col = train.columns[3]  # moisture_content
spec_cols  = train.columns[4:].tolist()

y_train     = train[target_col].values.astype(np.float64)
X_train_raw = train[spec_cols].values.astype(np.float64)
X_test_raw  = test[spec_cols].values.astype(np.float64)
test_ids    = test["sample number"].values
sp_train    = train["species number"].values

# ── Preprocessing functions ──
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

# ── LOSO-CV ──
def loso_folds(sp):
    for s in sorted(set(sp)):
        va = np.where(sp == s)[0]
        tr = np.where(sp != s)[0]
        yield tr, va, s

# ── P1 Baseline Pipeline ──
POWER = 0.27
LGBM_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

# 1. MSC (reference = train mean)
X_tr_msc, msc_ref = msc(X_train_raw)
X_te_msc, _ = msc(X_test_raw, reference=msc_ref)

# 2. SG derivative (window=9, poly=2, 1st deriv)
X_tr_sg = sg_deriv(X_tr_msc, window=9, polyorder=2, deriv=1)
X_te_sg = sg_deriv(X_te_msc, window=9, polyorder=2, deriv=1)

# 3. EPO (n=5, computed on all training data)
V_epo = compute_epo_matrix(X_tr_sg, y_train, sp_train, bin_width=10.0, n_components=5)
X_tr_epo = apply_epo(X_tr_sg, V_epo)
X_te_epo = apply_epo(X_te_sg, V_epo)

# 4. Target transform y^0.27
y_transformed = np.power(y_train, POWER)

# ===== EXPERIMENT MODIFICATION BELOW =====
# Implement noise augmentation for high moisture content samples.
# The noise will be added to the spectral data *before* EPO transformation.
# We'll target samples with y > 150% (assuming % means scaled value, adjust if necessary based on data distribution)
# A small standard deviation will be used for the noise.

NOISE_STD_HIGH_MC = 0.005  # Small standard deviation for noise
HIGH_MC_THRESHOLD = 150.0  # Threshold for high moisture content

# Identify high moisture content samples in the training set
high_mc_mask = y_train > HIGH_MC_THRESHOLD
X_train_high_mc = X_tr_epo[high_mc_mask]
y_train_high_mc = y_transformed[high_mc_mask]

# Generate noise for these samples
if X_train_high_mc.shape[0] > 0:
    noise = np.random.normal(0, NOISE_STD_HIGH_MC, X_train_high_mc.shape)
    X_train_augmented = X_tr_epo.copy()
    X_train_augmented[high_mc_mask] += noise
else:
    X_train_augmented = X_tr_epo.copy()

# LOSO-CV with augmented training data
oof = np.zeros(len(y_transformed))
test_preds = np.zeros(len(X_te_epo))
best_rounds = []

for tr_idx, va_idx, sp_id in loso_folds(sp_train):
    # Use the augmented training data for the current fold
    dtrain = lgb.Dataset(X_train_augmented[tr_idx], label=y_transformed[tr_idx])
    dval   = lgb.Dataset(X_train_augmented[va_idx], label=y_transformed[va_idx], reference=dtrain)
    model  = lgb.train(
        LGBM_PARAMS, dtrain, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof[va_idx] = model.predict(X_train_augmented[va_idx])
    test_preds += model.predict(X_te_epo) / len(set(sp_train))
    best_rounds.append(model.best_iteration)

# Inverse transform
oof_original = np.power(np.clip(oof, 0, None), 1.0 / POWER)
test_original = np.power(np.clip(test_preds, 0, None), 1.0 / POWER)

rmse = float(np.sqrt(np.mean((y_train - oof_original) ** 2)))
print(f"RMSE = {rmse:.4f}")
print(f"Avg best rounds: {int(np.mean(best_rounds))}")

# Save submission
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "pred": np.clip(test_original, 0, None)})
sub.to_csv(OUTPUT_DIR / f"submission_{EXP_ID}.csv", index=False, header=False)
print(f"Saved: {OUTPUT_DIR / f'submission_{EXP_ID}.csv'}")

# ===== END EXPERIMENT =====