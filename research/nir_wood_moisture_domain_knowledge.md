# Domain Knowledge: NIR Spectroscopy for Wood Moisture Content Prediction

## 1. Key Water Absorption Bands in NIR Spectra (Wavenumber Range 9993-3999 cm^-1)

### Primary Water Bands (most important for moisture prediction)

| Band Name | Wavenumber (cm^-1) | Wavelength (nm) | Molecular Vibration | Notes |
|-----------|-------------------|-----------------|---------------------|-------|
| **Combination band** | ~5200 cm^-1 (5280-5128) | ~1900-1950 nm | nu_as + delta(OH) asymmetric stretch + bending combination | **Strongest water band in this range**; most sensitive to moisture; intensity increases more rapidly with water concentration than overtone band |
| **1st Overtone** | ~6900 cm^-1 (7000-6200) | ~1430-1610 nm | nu_s + nu_as (OH) / 2*nu(OH) stretching overtone | Second strongest; broad; shifts with hydrogen bonding state |
| **2nd Overtone** | ~8333 cm^-1 | ~1200 nm | Higher order OH stretching | Weaker absorption |
| **3rd Overtone** | ~10300 cm^-1 | ~970 nm | Higher order OH stretching | Weakest; near edge of typical NIR range |

### Sub-band Details for Bound vs Free Water

| Feature | Wavenumber (cm^-1) | Description |
|---------|-------------------|-------------|
| Free OH stretching (overtone) | 7200-7000 | 2*nu(free OH); sharp peak |
| H-bonded OH (overtone) | 7000-6200 | 2*nu(OH) H-bonded; broader, red-shifted |
| H-bonded water (combination) | ~5280 | Hydrogen-bonded bulk water |
| Trapped/bound water (combination) | ~5190-5170 | Water trapped in polymer/cellulose matrix |
| Cellulose-bound water | 5180-5172 | Water strongly interacting with cellulose |

### Wood/Cellulose-Specific OH Bands

| Feature | Wavenumber (cm^-1) | Wavelength (nm) | Assignment |
|---------|-------------------|-----------------|------------|
| Cellulose OH | 7142-6250 | 1400-1600 | 1st harmonics of O-H stretching of cellulose + intramolecular H-bonds |
| Moisture-sensitive region | 7692-6900 | 1300-1450 | Overtones of OH due to moisture content |
| Wood moisture (combination) | 5190-5170 | 1927-1934 | Best for moisture amount analysis |
| Moisture visualization (hyperspectral) | 4464-5097 (=1966-2244 nm range) | 1966-2244 | Identified as optimal for MC distribution visualization |

### Practical Takeaway for Feature Engineering
- The **~5200 cm^-1 (1930-1950 nm)** and **~6900 cm^-1 (1440-1450 nm)** regions are THE most important for water/moisture prediction
- Regions 5190-5170 cm^-1 are especially suitable for moisture amount analysis
- The combination band (~5200) responds more linearly to moisture changes than the overtone band (~6900)
- In hydrophilic matrices like cellulose/wood, bound water shows a narrow, blue-shifted overtone band at ~7100 cm^-1

---

## 2. Wood Species Effects

### Key Findings
- **Moisture dominates NIR spectra of wood**: Moisture content exhibits strong absorption in NIR and can mask species-specific chemical differences
- **Global models can work**: A global PLS model across species achieved R^2_cv = 0.87, RMSECV = 1.08% for solid wood and R^2_cv = 0.85, RMSECV = 1.19% for wood powder
- **Species classification accuracy**: Global models correctly classified 85% of samples (benchtop FT-NIR) and 75% (portable devices)
- **Cross-section type matters more**: Models built on one section type (cross, radial, tangential) may not transfer well to other section types
- **EPO correction**: External Parameter Orthogonalization can correct moisture interference across different conditions

### Recommendations for Modeling
1. **Try global model first** with species as a categorical feature - often sufficient for moisture prediction
2. If accuracy is insufficient, consider **species-specific models** or species groupings
3. **EPO (External Parameter Orthogonalization)** is the best method to correct for moisture variation when predicting other wood properties
4. Wood anatomy (grain orientation / section type) may matter more than species

---

## 3. Preprocessing Techniques for NIR Spectroscopy Data

### Scatter Correction Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **SNV (Standard Normal Variate)** | Row-wise: subtract mean, divide by std of each spectrum | Corrects multiplicative scatter & path length; sample-by-sample; no reference spectrum needed |
| **MSC (Multiplicative Scatter Correction)** | Fits each spectrum to a reference (mean) spectrum via linear regression, corrects slope/offset | Similar to SNV but needs reference spectrum; better when spectra share a common baseline shape |
| **EMSC (Extended MSC)** | Extended version with polynomial baseline terms | More flexible; handles complex baselines; good for path-length correction |

### Derivative Methods

| Method | Description | Effect |
|--------|-------------|--------|
| **SG 1st Derivative** | Savitzky-Golay smoothing + 1st derivative | Removes additive baseline offset; sharpens overlapping bands; moderate noise amplification |
| **SG 2nd Derivative** | Savitzky-Golay smoothing + 2nd derivative | Removes linear baseline drift; further resolves overlapping bands; more noise amplification |
| **Norris-Williams Derivative** | Gap-segment derivative filter | Alternative to SG; simpler parameterization |

### SG Parameter Selection (Critical)
- **Window size**: Must be chosen carefully; too small = noisy, too large = over-smoothed
- **Polynomial order**: Typically 2 or 3; higher orders preserve peak shapes better
- **Derivative order**: 1st or 2nd most common; 2nd derivative inverts and sharpens peaks

### Recommended Preprocessing Combinations
1. **SNV + SG 2nd derivative**: Best overall combination in many studies; R^2 = 0.81 (calibration), R^2 = 0.79 (cross-validation) for wood moisture
2. **SNV alone**: Best for visualizing MC distribution
3. **EMSC + 1st derivative**: Best for estimating and visualizing moisture content in the 1966-2244 nm region
4. **SG smoothing + SNV**: Performed better than other preprocessing functions across 13 food/plant datasets

### Other Preprocessing
- **Wavelet Transform (WT)**: Alternative denoising approach
- **Mean centering**: Standard; subtract column means
- **Autoscaling**: Mean center + divide by column std
- **Detrending**: Remove polynomial baseline trends

---

## 4. Best ML Models for NIR Chemometrics

### Model Comparison Summary

| Model | Strengths | Weaknesses | When to Use |
|-------|-----------|------------|-------------|
| **PLS (Partial Least Squares)** | Gold standard; handles multicollinearity; interpretable; works with small samples; fast | Linear only; needs good preprocessing | Default choice; <2000 samples; interpretability needed |
| **PCR (Principal Component Regression)** | Similar to PLS but uses PCA | Ignores Y in component selection; generally slightly worse than PLS | Rarely preferred over PLS |
| **SVM (Support Vector Machine)** | Handles nonlinearity (RBF kernel); 14-29% lower RMSEP than PLS on large datasets; R^2 = 0.91 | Requires hyperparameter tuning; less interpretable | Large datasets (>1000 samples); nonlinear relationships |
| **Random Forest** | Handles nonlinearity; feature importance built-in; robust to outliers | Can overfit; many hyperparameters | Medium-large datasets; variable importance needed |
| **CNN (Convolutional Neural Network)** | Learns features automatically; handles spatial/spectral patterns | Requires >2000 samples; black box; computationally expensive | Large datasets; complex nonlinear patterns |
| **ELM (Extreme Learning Machine)** | Fast training; good with limited samples | Less stable; random initialization | Limited samples; fast prototyping |

### Decision Framework
- **< 500 samples**: PLS is king; SVM may help marginally
- **500-2000 samples**: PLS + wavelength selection; SVM with RBF kernel competitive
- **> 2000 samples**: CNN and deep learning can outperform PLS; SVM still strong
- **Nonlinear relationships (e.g., high moisture content)**: SVM or ensemble methods preferred
- **Best practice**: Always benchmark against PLS first

### Performance Benchmarks for Wood Moisture
- PLS with NIR: R^2 = 0.89-0.98, RMSEP = 5.1-18.3% (varies by study/wood type)
- Combined SNV + SG + PLS: R^2 = 0.81 (calibration), R^2 = 0.79 (cross-validation)

---

## 5. Wavelength/Variable Selection Strategies

### Most Effective Methods

| Method | Abbreviation | Mechanism | Notes |
|--------|-------------|-----------|-------|
| **Competitive Adaptive Reweighted Sampling** | CARS | "Survival of the fittest" - iteratively selects wavelengths with large PLS regression coefficients; cross-validation selects optimal set | Very popular; Darwin-inspired; good first choice |
| **Successive Projections Algorithm** | SPA | Projects candidates onto orthogonal subspaces; selects vars with largest projection norms | Minimizes multicollinearity; complementary to CARS |
| **Iteratively Retains Informative Variables** | IRIV | Exhaustive approach using Mann-Whitney U test to determine optimal features | IRIV outperforms CARS and SPA overall |
| **Variable Importance in Projection** | VIP | PLS-derived importance scores; VIP > 1 typically considered important | Simple; built into PLS; good screening tool |
| **Genetic Algorithm** | GA | Evolutionary optimization of wavelength subsets | Improved apple quality assessment by 30% |
| **Bootstrapping Soft Shrinkage** | BOSS | Demonstrated superiority in selecting instructive wavenumbers | Newer method |
| **Interval PLS** | iPLS / SiPLS | Splits spectrum into intervals; finds optimal intervals or synergy intervals | Good for identifying informative regions |

### Hybrid Strategies (Best Performance)
- **SiPLS-CARS**: R^2 > 0.99 reported (better than single methods)
- **CARS-IRIV-PLS**: Effective combination
- **BOSS-IRIV-PLS/ELM**: Task-specific optimization
- **EPO-IRIV-PLS**: Best when moisture correction + feature selection both needed
- **Three-step strategies**: Coarse screening -> fine selection -> final optimization

### Practical Recommendations
1. Start with **VIP scores from PLS** to identify obviously important regions
2. Apply **CARS** or **IRIV** for refined selection
3. Consider **SPA** to reduce multicollinearity among selected wavelengths
4. Hybrid methods (e.g., CARS+SPA, SiPLS+CARS) generally outperform single methods
5. Always validate that selected wavelengths correspond to known chemical bands (see Section 1)

---

## 6. Known Issues and Challenges

### Baseline Drift
- **Causes**: Instrumental fluctuation, scattering variations, sample constitution changes, temperature-unstabilized CCD detector
- **Solutions**: SNV, MSC, derivatives (1st derivative removes offset, 2nd removes linear drift), detrending

### Scattering Effects
- **Causes**: Particle size variation, sample packing density, matrix inhomogeneity, surface roughness
- **Effects**: Multiplicative and additive distortions that obscure analyte signal
- **Solutions**: SNV (row-wise), MSC/EMSC (model-based), derivatives

### Temperature Effects (Critical for Wood)
- **Hydroxyl band shifts**: The two main OH bands at ~1450 nm and ~1930 nm shift by ~0.4 nm/degree C toward shorter wavelengths with increasing temperature
- **Freezing point discontinuity**: Marked peak shifts of ~25 nm (band A) and ~20 nm (band B) when water in wood freezes/thaws at 0 degrees C; especially problematic for sapwood
- **Solutions**: Temperature-controlled measurements; include temperature in model; temperature-robust preprocessing; global models trained across temperature range

### Nonlinearity at High Moisture Content
- **Cause**: Beer-Lambert law breaks down at high absorbance; saturation effects; strong hydrogen bonding changes
- **The combination band (~5200 cm^-1) intensity increases more rapidly** than the overtone band with water concentration - different bands may show different linearity ranges
- **Solutions**: Nonlinear models (SVM, RF); piecewise linear models; restrict calibration range; use less absorbing bands (overtone vs combination)

### Other Issues
- **Detector nonlinearity and stray light**: Instrumental artifacts
- **Wavelength misalignment**: Between instruments or over time
- **Sample morphology**: Wood grain direction, section type (cross/radial/tangential) significantly affects spectra
- **Model transfer**: Models built on one instrument/condition may not transfer well to another; standardization techniques needed

---

## Summary: Recommended Pipeline for Wood Moisture from NIR

1. **Preprocessing**: SNV + SG 2nd derivative (or 1st derivative), window size tuned via CV
2. **Initial model**: PLS regression as baseline
3. **Wavelength selection**: VIP screening -> CARS or IRIV refinement -> validate against known water bands
4. **Advanced models**: If PLS insufficient, try SVM (RBF kernel) or ensemble methods
5. **Species handling**: Start with global model + species feature; split if needed
6. **Key wavelengths to prioritize**: ~5200 cm^-1 region AND ~6900 cm^-1 region
7. **Validation**: Always use independent test set; watch for temperature/moisture range coverage
