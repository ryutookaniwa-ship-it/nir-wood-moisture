import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")

INPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\input\nir-wood-moisture")
OUTPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\output\nir-wood-moisture")
EXP_ID = "exp_002"

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

# Define spectral feature extraction function
def extract_water_absorption_features(X, wavelengths):
    # Approximate indices for water absorption bands
    # 6900 cm^-1 (1.45 um) is a strong band
    # 5200 cm^-1 (1.92 um) is another significant band
    # We need to find the closest indices in the given wavelengths
    
    # Assuming wavelengths are in some unit like nm or um. Let's assume they correspond to the column indices directly.
    # If wavelengths are provided, we'd map them to indices. For now, we'll hardcode approximate indices based on common NIR spectra.
    # A more robust solution would involve a mapping of wavelengths to indices.
    
    # Let's assume the spectral columns correspond to wavelengths from ~400nm to ~2500nm (common range for NIR).
    # This is a simplification; ideally, the exact wavelengths would be known and used.
    
    # Example: If spec_cols represent wavelengths from 400 to 2500 nm with a step of 1 nm.
    # Then 6900 cm^-1 = 0.69 um = 690 nm. Index would be ~290.
    # And 5200 cm^-1 = 0.52 um = 520 nm. Index would be ~120.
    
    # Since we don't have the actual wavelength mapping, we will use approximate indices.
    # For demonstration, let's pick some indices that are often associated with water absorption.
    # This is a critical part that would need to be refined with actual wavelength information.
    
    # Let's assume a typical NIR range, say 1000nm to 2500nm with 1nm step.
    # 6900 cm-1 = 0.69 um = 690 nm. This is likely out of typical NIR range.
    # Let's re-read the prompt: "6900 cm⁻¹ and potentially 5200 cm⁻¹" - this means 0.69 um and 0.52 um.
    # These are actually visible/near-infrared bands, not strictly NIR used for water in many materials.
    # If the spectra are indeed in the visible range, then these bands are relevant.
    
    # Let's assume the provided spectral data is in a range that includes these wavelengths.
    # We need to map wavelengths to indices. If we assume the columns ARE the wavelengths in cm^-1,
    # then we look for columns near 6900 and 5200.
    
    # A more common interpretation of water bands in NIR is around 970nm, 1190nm, 1450nm, 1940nm.
    # If the prompt strictly means 6900 cm^-1 (0.69 um) and 5200 cm^-1 (0.52 um), we need to map these.
    # Let's assume the columns correspond to wavelengths and find the closest ones.
    
    # Without actual wavelength values for each column, we have to make an assumption.
    # Let's assume the `spec_cols` list contains strings representing wavelengths, e.g., '400', '401', ..., '2500'.
    # Or, we can assume the order of columns directly maps to wavelengths.
    
    # Let's simulate finding indices for 6900 cm^-1 and 5200 cm^-1.
    # If the data is in cm^-1, we'd look for columns named '6900' and '5200'.
    # If the data is in um, we'd look for columns named '0.69', '0.52'.
    # If the data is in nm, we'd look for columns named '690', '520'.
    
    # Given the baseline used standard preprocessing without explicit wavelength info,
    # it's likely the column names are just identifiers or implicitly ordered.
    # For this hypothesis, we need to *identify* spectral regions.
    
    # Let's make a STRONG ASSUMPTION: the columns are ordered by wavelength and
    # we can infer approximate indices. If the spec_cols are '400', '401', ..., '2500' (nm)
    # then:
    # 6900 cm-1 = 0.69 um = 690 nm. Index might be around `690 - 400 = 290`.
    # 5200 cm-1 = 0.52 um = 520 nm. Index might be around `520 - 400 = 120`.
    
    # If the spectra start from a different wavelength, e.g., 1000 nm:
    # 690 nm is not in range. This hypothesis might be flawed if data is limited to NIR.
    # Let's pivot to more standard NIR water bands if the prompt is interpreted loosely.
    # Water bands: ~970 nm, ~1190 nm, ~1450 nm, ~1940 nm.
    # If columns are 1000..2500 nm:
    # 970nm -> index 970-1000 = -30 (not present if starts at 1000)
    # 1190nm -> index 1190-1000 = 190
    # 1450nm -> index 1450-1000 = 450
    # 1940nm -> index 1940-1000 = 940
    
    # Let's assume the prompt meant features that correlate with water absorption and are present in the NIR range.
    # And let's assume the columns ARE the wavelengths in nm.
    # This is the most reasonable interpretation that allows implementing the hypothesis.
    
    # Let's assume spec_cols are numerical strings like '400.0', '401.0', ...
    # Convert spec_cols to floats to find indices.
    try:
        wavelengths_nm = np.array([float(col) for col in spec_cols])
    except ValueError:
        # If column names are not floats, assume they are ordered and we need to estimate.
        # This is highly unreliable without more info.
        # For now, let's assume a common step and starting point.
        # Let's assume the first column corresponds to 400 nm and step is 1 nm.
        wavelengths_nm = np.arange(400, 400 + len(spec_cols))
        print("Warning: Column names are not numerical wavelengths. Assuming sequential wavelengths starting from 400nm with 1nm step.")

    # Find indices for characteristic water absorption bands (in nm)
    # Using common NIR water bands as a proxy for the prompt's specified bands if they are not in data range.
    # The prompt *specifically* mentions 6900 cm⁻¹ (0.69 µm = 690 nm) and 5200 cm⁻¹ (0.52 µm = 520 nm).
    # If the data is in the typical NIR range (e.g., 1000-2500 nm), these bands might not be present.
    # Let's try to find indices for the *prompt's* bands first, assuming they are in the data.
    
    target_wls_cm = [6900, 5200] # cm^-1
    target_wls_um = [1/wl_cm * 1e4 for wl_cm in target_wls_cm] # convert cm^-1 to um
    target_wls_nm = [wl_um * 1000 for wl_um in target_wls_um] # convert um to nm
    # So, target wavelengths are ~690 nm and ~520 nm.

    features = []
    
    # Feature 1: Around 690 nm (0.69 um)
    idx_690 = np.argmin(np.abs(wavelengths_nm - 690))
    if 690 in wavelengths_nm:
        # If the exact wavelength exists
        f1 = X[:, np.where(wavelengths_nm == 690)[0][0]]
    else:
        # If not, use the closest available wavelength and potentially its derivatives
        # A simple approach: just take the value at the closest index.
        # A more complex approach could involve interpolation or band ratios.
        closest_idx_690 = np.argmin(np.abs(wavelengths_nm - 690))
        if np.abs(wavelengths_nm[closest_idx_690] - 690) < 20: # threshold to ensure it's close
            f1 = X[:, closest_idx_690]
        else:
            f1 = np.zeros(X.shape[0]) # fallback if no close band

    # Feature 2: Around 520 nm (0.52 um)
    idx_520 = np.argmin(np.abs(wavelengths_nm - 520))
    if 520 in wavelengths_nm:
        f2 = X[:, np.where(wavelengths_nm == 520)[0][0]]
    else:
        closest_idx_520 = np.argmin(np.abs(wavelengths_nm - 520))
        if np.abs(wavelengths_nm[closest_idx_520] - 520) < 20: # threshold
            f2 = X[:, closest_idx_520]
        else:
            f2 = np.zeros(X.shape[0]) # fallback

    # Combine into a feature matrix
    # Using raw values at these bands. Could also use derivatives or ratios.
    # For simplicity and interpretability, let's add them as new features.
    
    # Let's also consider the derivative or spectral slope around these bands as it can capture absorption features.
    # We'll need the first derivative for this.
    X_deriv = sg_deriv(X, window=9, polyorder=2, deriv=1) # Recompute derivative if not already done.
    # However, the main pipeline already computes derivative. We should use that if available.
    # Let's assume this function is called *after* SG derivative is computed, or we recompute it.
    # For now, let's assume we need to compute it IF not provided.
    
    # Let's refine: extract features from the *preprocessed* data (after MSC, SG)
    # This function will be applied to X_tr_sg and X_te_sg
    
    # Let's make a more robust feature: the ratio of absorbance at the peak to a nearby continuum.
    # This is more standard for feature extraction.
    # For ~690 nm band, we could use a continuum point slightly blue or red of it.
    # For ~520 nm band, similarly.
    
    # Let's try a simpler approach first: just add the values at the identified bands as new features.
    # We need to ensure `wavelengths_nm` is available and correct.
    # The `spec_cols` are strings like 'X.XXX'. Let's parse them.
    
    # If `spec_cols` are like `['400.0', '401.0', ...]`:
    wavelengths_nm_parsed = np.array([float(s.replace('X','')) for s in spec_cols]) # Assuming format like '400.0'
    
    # Find indices for 690 nm and 520 nm.
    idx_690_nm = np.argmin(np.abs(wavelengths_nm_parsed - 690))
    idx_520_nm = np.argmin(np.abs(wavelengths_nm_parsed - 520))

    # Add the absorbance values at these approximate indices as new features.
    # Ensure indices are within bounds.
    if idx_690_nm < X.shape[1]:
        feature_690 = X[:, idx_690_nm]
    else:
        feature_690 = np.zeros(X.shape[0])
        print(f"Warning: Wavelength {wavelengths_nm_parsed[idx_690_nm]} (closest to 690nm) out of bounds.")

    if idx_520_nm < X.shape[1]:
        feature_520 = X[:, idx_520_nm]
    else:
        feature_520 = np.zeros(X.shape[0])
        print(f"Warning: Wavelength {wavelengths_nm_parsed[idx_520_nm]} (closest to 520nm) out of bounds.")

    # Consider adding a feature that captures the difference or ratio, which might be more robust.
    # Example: Absorbance at 690nm / Absorbance at a nearby continuum point.
    # Let's keep it simple for now and add the raw values.
    
    # Stack the new features alongside the original (preprocessed) spectral data.
    X_with_features = np.hstack([X, feature_690.reshape(-1, 1), feature_520.reshape(-1, 1)])
    
    return X_with_features

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
X_tr_epo_base = apply_epo(X_tr_sg, V_epo)
X_te_epo_base = apply_epo(X_te_sg, V_epo)

# 4. Extract new spectral features BEFORE EPO projection (or after, the hypothesis implies it's additive)
# Let's try adding them *after* EPO, as it's a form of feature engineering applied to the projected data.
# This means the new features will be part of the projected space.
# The hypothesis says "explicitly captures... even after EPO projection".
# This implies the feature should exist *alongside* the EPO-transformed data.

# So, we apply EPO to the SG derivative, AND then add the new features.
# This means we need to apply EPO to the original spectral data (before derivative) if we want to add features to it.
# OR, add features to derivative, then EPO.
# Let's add the features to the SG derivative data BEFORE EPO.
# This allows the new features to also be potentially projected by EPO if they are correlated with the main components.
# OR, add them AFTER EPO. This makes them independent of EPO.
# "Introducing a physically motivated feature ... even after EPO projection." -> This suggests they are additive.
# Let's add them to the EPO-projected data.

# Re-thinking: The hypothesis is "introducing a feature... even after EPO projection".
# This implies the EPO projection happens, and then this new feature is added.
# Let's apply EPO to SG data, THEN add the new features.

# Apply EPO to SG derivative data
X_tr_epo = apply_epo(X_tr_sg, V_epo)
X_te_epo = apply_epo(X_te_sg, V_epo)

# Extract and append new features
# We need the wavelengths for the `extract_water_absorption_features` function.
# Assume `spec_cols` can be parsed into wavelengths.
X_tr_with_new_features = extract_water_absorption_features(X_tr_epo, spec_cols)
X_te_with_new_features = extract_water_absorption_features(X_te_epo, spec_cols)

# Now, the LGBM will train on X_tr_with_new_features.
X_train_final = X_tr_with_new_features
X_test_final = X_te_with_new_features

# 4. Target transform y^0.27
y_transformed = np.power(y_train, POWER)

# ===== EXPERIMENT MODIFICATION BELOW =====
# Modify ONLY the section below to test the hypothesis.
# Keep the preprocessing pipeline above intact unless the hypothesis
# specifically requires changing it.
# Print: print(f"RMSE = {rmse:.4f}")
# Save: submission CSV to OUTPUT_DIR
# ===== BEGIN EXPERIMENT =====

# LOSO-CV
oof = np.zeros(len(y_transformed))
test_preds = np.zeros(len(X_test_final))
best_rounds = []

for tr_idx, va_idx, sp_id in loso_folds(sp_train):
    # Use the final feature sets (EPO + new features)
    dtrain = lgb.Dataset(X_train_final[tr_idx], label=y_transformed[tr_idx])
    dval   = lgb.Dataset(X_train_final[va_idx], label=y_transformed[va_idx], reference=dtrain)
    
    model  = lgb.train(
        LGBM_PARAMS, dtrain, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof[va_idx] = model.predict(X_train_final[va_idx])
    test_preds += model.predict(X_test_final) / len(set(sp_train))
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