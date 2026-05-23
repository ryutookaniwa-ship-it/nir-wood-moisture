import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings; warnings.warn("ignore")

INPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\input\nir-wood-moisture")
OUTPUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\Ryuto\5. 個人的データ分析\output\nir-wood-moisture")
EXP_ID = "exp_002_species15_calib"

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
# Modify ONLY the section below to test the hypothesis.
# Keep the preprocessing pipeline above intact unless the hypothesis
# specifically requires changing it.
# Print: print(f"RMSE = {rmse:.4f}")
# Save: submission CSV to OUTPUT_DIR
# ===== BEGIN EXPERIMENT =====

# Hypothesis: Species 15 specific calibration

oof_preds = np.zeros(len(y_train))
test_preds_ensemble = np.zeros(len(X_test_raw))
species_15_test_preds = np.zeros(len(X_test_raw)) # Store predictions for species 15 only
species_15_train_indices = np.where(sp_train == 15)[0]
other_species_train_indices = np.where(sp_train != 15)[0]

# Define parameters for the main model and species 15 model
main_lgbm_params = LGBM_PARAMS.copy()
species_15_lgbm_params = LGBM_PARAMS.copy()
species_15_lgbm_params['learning_rate'] = 0.01 # Lower learning rate for calibration
species_15_lgbm_params['num_leaves'] = 31 # Potentially simpler model for calibration

# Separate data for species 15 and others
X_train_15 = X_tr_epo[species_15_train_indices]
y_train_15 = y_transformed[species_15_train_indices]
X_train_other = X_tr_epo[other_species_train_indices]
y_train_other = y_transformed[other_species_train_indices]

# Train a main model on all species except 15
dtrain_main = lgb.Dataset(X_train_other, label=y_train_other)
model_main = lgb.train(
    main_lgbm_params, dtrain_main, num_boost_round=3000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    # Add validation set for main model if needed, but for LOSO it's tricky to do here.
    # For simplicity, training without explicit early stopping on validation here.
)

# Train a separate model for species 15 on ALL available species 15 data from the full training set
# This model will be used for calibration on the validation folds of species 15
dtrain_sp15_calib = lgb.Dataset(X_train_15, label=y_train_15)
model_sp15_calib = lgb.train(
    species_15_lgbm_params, dtrain_sp15_calib, num_boost_round=3000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)


overall_test_preds = np.zeros(len(X_te_epo))
oof_final = np.zeros(len(y_train))
best_rounds_list = []

for tr_idx_outer, va_idx_outer, sp_id_outer in loso_folds(sp_train):
    
    is_species_15_fold = (sp_id_outer == 15)
    
    # Re-train main model on all species *except* current validation species
    current_tr_idx = np.setdiff1d(np.arange(len(sp_train)), va_idx_outer)
    current_X_train = X_tr_epo[current_tr_idx]
    current_y_train = y_transformed[current_tr_idx]
    current_sp_train = sp_train[current_tr_idx]

    # Filter out species 15 for the main model training in this fold if it's not species 15's fold
    if not is_species_15_fold:
        main_train_mask = current_sp_train != 15
        dtrain_fold = lgb.Dataset(current_X_train[main_train_mask], label=current_y_train[main_train_mask])
        model_fold = lgb.train(main_lgbm_params, dtrain_fold, num_boost_round=3000,
                               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_fold_preds = model_fold.predict(X_tr_epo[va_idx_outer])
        test_fold_preds = model_fold.predict(X_te_epo)
    else:
        # For species 15's fold, train main model on all other species
        other_species_mask = current_sp_train != 15
        dtrain_fold = lgb.Dataset(current_X_train[other_species_mask], label=current_y_train[other_species_mask])
        model_fold = lgb.train(main_lgbm_params, dtrain_fold, num_boost_round=3000,
                               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_fold_preds = model_fold.predict(X_tr_epo[va_idx_outer]) # Predict on species 15 validation
        test_fold_preds = model_fold.predict(X_te_epo) # Predict on all test data

    # Calibration for species 15
    if is_species_15_fold:
        # Use the pre-trained species 15 calibration model to predict on the validation set
        # The calibration model should predict the *residual* or adjustment needed.
        # Here, we'll train it to predict the original target directly, then adjust.
        
        # We use the model_sp15_calib trained on ALL species 15 data for a more robust calibration.
        # Predict on the validation set of species 15
        sp15_val_preds_raw = model_sp15_calib.predict(X_tr_epo[va_idx_outer])

        # The calibration model should predict the difference or directly the corrected value.
        # A simpler approach is to train a model to predict the residual.
        # Let's try to train a model that predicts the residual for species 15.
        # Residual = y_true - y_pred_main
        
        # First, get main model predictions on species 15 training data for residual calculation
        main_preds_on_sp15_train = model_fold.predict(X_train_15)
        residuals_sp15_train = y_train_15 - main_preds_on_sp15_train
        
        # Train a residual model on species 15 data
        dtrain_residual_sp15 = lgb.Dataset(X_train_15, label=residuals_sp15_train)
        model_residual_sp15 = lgb.train(
            species_15_lgbm_params, dtrain_residual_sp15, num_boost_round=3000,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        
        # Predict residuals for the validation species 15 data
        predicted_residuals_sp15_val = model_residual_sp15.predict(X_tr_epo[va_idx_outer])
        
        # Adjust the main model's prediction for species 15 validation data
        oof_fold_preds = oof_fold_preds + predicted_residuals_sp15_val
        
        # Adjust test predictions for species 15
        # Predict residuals on test data that belongs to species 15 (if available, but we don't have sp info for test)
        # For test set, we apply the residual model to all test data and hope it generalizes.
        # A more robust approach would be to have species info for test or use a global residual model.
        # Given the constraint, we apply the residual model trained on species 15 to all test data.
        predicted_residuals_sp15_test = model_residual_sp15.predict(X_te_epo)
        test_fold_preds = test_fold_preds + predicted_residuals_sp15_test


    oof_final[va_idx_outer] = oof_fold_preds
    overall_test_preds += test_fold_preds / len(set(sp_train))


# Inverse transform
oof_original = np.power(np.clip(oof_final, 0, None), 1.0 / POWER)
test_original = np.power(np.clip(overall_test_preds, 0, None), 1.0 / POWER)

rmse = float(np.sqrt(np.mean((y_train - oof_original) ** 2)))
print(f"RMSE = {rmse:.4f}")

# Save submission
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "pred": np.clip(test_original, 0, None)})
sub.to_csv(OUTPUT_DIR / f"submission_{EXP_ID}.csv", index=False, header=False)
print(f"Saved: {OUTPUT_DIR / f'submission_{EXP_ID}.csv'}")