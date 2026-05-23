import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src/nir")

from nir_loso_utils import (
    load_data, msc, snv, sg_deriv,
    loso_folds, loso_rmse, loso_lgbm, loso_sklearn,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb