# NIR Wood Moisture Prediction

Predicting wood moisture content from Near-Infrared (NIR) spectroscopy data.
**Signate Competition** | Best LB RMSE: **15.395**

## Problem

Given NIR spectra (1555 wavenumbers, 4000-10000 cm-1) of wood samples across 19 species, predict moisture content (%). The challenge: **training and test species are completely disjoint** - the model must generalize to unseen species.

## Key Findings

1. **LOSO-CV is essential**: Standard KFold leaks species-specific patterns. Leave-One-Species-Out CV correctly estimates generalization error.
2. **EPO (External Parameter Orthogonalization)** removes species-dependent spectral variation while preserving moisture signal. Improved LB from 18.4 to 16.1.
3. **Power transform (y^0.27)** compresses high-moisture predictions, reducing RMSE by 0.7.
4. **LightGBM + feature_fraction=0.07** acts as a random subspace method, preventing species-specific memorization.
5. **Neural networks (MLP, 1D-CNN, Transformer) all failed** - inverse correlation between LOSO-CV and LB score. Tree models with random subspace are fundamentally better for small-sample spectral data.

## Pipeline

```
Raw Spectra (1555d)
  -> MSC (Multiplicative Scatter Correction)
  -> Savitzky-Golay (w=9, poly=2, 1st derivative)
  -> EPO (n_components=5, bin_width=10)
  -> LightGBM (leaves=63, mcs=10, ff=0.07, lr=0.02)
  -> Inverse power transform
```

## Score Trajectory

| Experiment | Approach | LOSO-RMSE | LB-RMSE |
|-----------|----------|-----------|---------|
| G | Raw + LGBM baseline | 21.48 | 18.995 |
| R | MSC + SG(w=9) + sqrt | 19.68 | 18.403 |
| B2 | + EPO(n=5) | 16.44 | 17.651 |
| I2 | SG(w=9,p=2) + EPO | 15.73 | **16.101** |
| M2 | + Power transform (p=0.30) | 15.59 | 15.545 |
| **P1** | **+ Fine-tuned p=0.27** | **15.47** | **15.395** |

## Tech Stack

- Python, LightGBM, scikit-learn, NumPy, SciPy
- Custom EPO implementation for spectral preprocessing
- LOSO cross-validation framework

## Project Structure

```
src/           # Experiment scripts (100+)
scores/        # Score tracking & experiment log
docs/          # Domain knowledge & analysis
research/      # Literature review
input/         # Competition data (not tracked)
output/        # Model outputs (not tracked)
```

## Lessons Learned

- In chemometrics, **domain-specific preprocessing > hyperparameter tuning**
- EPO is the single most impactful technique (+2.5 RMSE improvement)
- CV strategy matters more than model choice
- Ensemble fails when base models have r > 0.98 correlation
