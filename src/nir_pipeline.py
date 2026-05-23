"""
NIR Wood Moisture - Self-contained P1 Pipeline Module
=====================================================
Night agent / experiment scripts import this to avoid code duplication.
All paths are relative to the project root.

Current best: P1 (LB-RMSE = 15.395)
Pipeline: MSC -> SG(w=9, p=2, d=1) -> EPO(n=5, bw=10) -> LGBM(y^0.27)
"""
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCORES_DIR = PROJECT_ROOT / "scores"
LOGS_DIR = OUTPUT_DIR / "logs"

# ── P1 Best Parameters ─────────────────────────────────────────────────────
P1_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
}

P1_POWER = 0.27
P1_SG_WINDOW = 9
P1_SG_POLYORDER = 2
P1_SG_DERIV = 1
P1_EPO_N_COMPONENTS = 5
P1_EPO_BIN_WIDTH = 10.0


# ── Data Loading ────────────────────────────────────────────────────────────
def load_data():
    """Load train/test data. Returns dict with arrays."""
    train = pd.read_csv(INPUT_DIR / "train.csv", encoding="shift-jis")
    test = pd.read_csv(INPUT_DIR / "test.csv", encoding="shift-jis")

    target_col = train.columns[3]
    spec_cols = train.columns[4:].tolist()

    return {
        "y_train": train[target_col].values.astype(np.float64),
        "X_train_raw": train[spec_cols].values.astype(np.float64),
        "X_test_raw": test[spec_cols].values.astype(np.float64),
        "sp_train": train["species number"].values,
        "test_ids": test.iloc[:, 0].values,
        "spec_cols": spec_cols,
        "wavenumbers": np.array([float(c) for c in spec_cols]),
    }


# ── Preprocessing ───────────────────────────────────────────────────────────
def msc(X, reference=None):
    """Multiplicative Scatter Correction."""
    ref = reference if reference is not None else X.mean(axis=0)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        coef = np.polyfit(ref, X[i], 1)
        out[i] = (X[i] - coef[1]) / coef[0]
    return out, ref


def sg_deriv(X, window=9, polyorder=2, deriv=1):
    """Savitzky-Golay derivative."""
    return savgol_filter(
        X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1
    )


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    """Compute EPO projection matrix for species-effect removal."""
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 4:
            continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species:
            continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1:
            continue
        pca = PCA(n_components=n_c, random_state=42)
        pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs:
        return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T


def apply_epo(X, V):
    """Apply EPO projection."""
    return X - (X @ V) @ V.T


# ── CV Framework ────────────────────────────────────────────────────────────
def loso_folds(sp):
    """Leave-One-Species-Out cross-validation folds."""
    for s in sorted(set(sp)):
        va = np.where(sp == s)[0]
        tr = np.where(sp != s)[0]
        yield tr, va, s


def loso_rmse(oof_preds, y_true):
    """Compute RMSE from OOF predictions."""
    return float(np.sqrt(np.mean((y_true - oof_preds) ** 2)))


def per_species_rmse(oof_preds, y_true, sp):
    """Compute RMSE per species. Returns dict {species_id: rmse}."""
    result = {}
    for s in sorted(set(sp)):
        mask = sp == s
        rmse = float(np.sqrt(np.mean((y_true[mask] - oof_preds[mask]) ** 2)))
        result[s] = rmse
    return result


# ── P1 Pipeline (Full) ──────────────────────────────────────────────────────
def p1_preprocess(X_train_raw, X_test_raw, y_train, sp_train,
                  sg_window=P1_SG_WINDOW, sg_polyorder=P1_SG_POLYORDER,
                  sg_deriv_order=P1_SG_DERIV,
                  epo_n=P1_EPO_N_COMPONENTS, epo_bw=P1_EPO_BIN_WIDTH):
    """Apply full P1 preprocessing pipeline. Returns (X_train, X_test, metadata)."""
    # MSC
    X_tr_msc, msc_ref = msc(X_train_raw)
    X_te_msc, _ = msc(X_test_raw, reference=msc_ref)

    # SG derivative
    X_tr_sg = sg_deriv(X_tr_msc, window=sg_window, polyorder=sg_polyorder, deriv=sg_deriv_order)
    X_te_sg = sg_deriv(X_te_msc, window=sg_window, polyorder=sg_polyorder, deriv=sg_deriv_order)

    # EPO
    V = compute_epo_matrix(X_tr_sg, y_train, sp_train,
                           bin_width=epo_bw, n_components=epo_n)
    X_tr = apply_epo(X_tr_sg, V)
    X_te = apply_epo(X_te_sg, V)

    meta = {
        "msc_ref": msc_ref,
        "epo_V": V,
        "sg_window": sg_window,
        "sg_polyorder": sg_polyorder,
        "epo_n": epo_n,
        "epo_bw": epo_bw,
    }
    return X_tr, X_te, meta


def p1_train_eval(X_tr, y_train, sp_train, params=None, power=P1_POWER,
                  num_boost_round=3000, early_stopping_rounds=50):
    """Train with LOSO-CV and return OOF predictions + per-fold iterations."""
    if params is None:
        params = P1_PARAMS.copy()

    y_trans = np.power(y_train, power)
    oof_trans = np.zeros(len(y_train))
    iters = []
    sp_rmses = {}

    for tr_idx, va_idx, sp_id in loso_folds(sp_train):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_trans[tr_idx])
        dval = lgb.Dataset(X_tr[va_idx], label=y_trans[va_idx], reference=dtrain)
        model = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_trans[va_idx] = model.predict(X_tr[va_idx])
        iters.append(model.best_iteration)

    # Inverse transform
    oof = np.power(np.clip(oof_trans, 0, None), 1.0 / power)
    rmse = loso_rmse(oof, y_train)
    sp_rmses = per_species_rmse(oof, y_train, sp_train)

    return {
        "oof": oof,
        "oof_trans": oof_trans,
        "rmse": rmse,
        "sp_rmses": sp_rmses,
        "avg_iter": int(np.mean(iters)),
        "iters": iters,
    }


def p1_predict(X_tr, X_te, y_train, params=None, power=P1_POWER, avg_iter=None):
    """Train on full data and predict test set."""
    if params is None:
        params = P1_PARAMS.copy()

    y_trans = np.power(y_train, power)
    dtrain = lgb.Dataset(X_tr, label=y_trans)

    if avg_iter is None:
        avg_iter = 700  # fallback

    model = lgb.train(
        params, dtrain, num_boost_round=avg_iter,
        callbacks=[lgb.log_evaluation(-1)],
    )
    preds_trans = model.predict(X_te)
    preds = np.power(np.clip(preds_trans, 0, None), 1.0 / power)
    return preds


# ── Submission ──────────────────────────────────────────────────────────────
def save_submission(test_ids, preds, filepath):
    """Save submission CSV (no header, 2 columns)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": test_ids, "pred": preds})
    df.to_csv(filepath, index=False, header=False)
    print(f"Saved: {filepath}")


# ── Experiment Logging ──────────────────────────────────────────────────────
def log_experiment(exp_id, hypothesis, result, analysis, next_hypothesis=None):
    """Log experiment result to JSON session file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{timestamp}_night_session.json"

    entry = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "hypothesis": hypothesis,
        "score": result.get("rmse") if isinstance(result, dict) else None,
        "delta_vs_p1": (result.get("rmse", 0) - 15.4725) if isinstance(result, dict) else None,
        "avg_iter": result.get("avg_iter") if isinstance(result, dict) else None,
        "sp_rmses": {str(k): round(v, 2) for k, v in result.get("sp_rmses", {}).items()} if isinstance(result, dict) else None,
        "analysis": analysis,
        "next_hypothesis": next_hypothesis,
    }

    # Append to existing session or create new
    if log_path.exists():
        with open(log_path) as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Logged: {log_path}")
    return log_path


# ── Convenience: Run P1 Baseline ────────────────────────────────────────────
def run_p1_baseline():
    """Run the full P1 pipeline and return results. Useful for verification."""
    data = load_data()
    X_tr, X_te, meta = p1_preprocess(
        data["X_train_raw"], data["X_test_raw"],
        data["y_train"], data["sp_train"]
    )
    result = p1_train_eval(X_tr, data["y_train"], data["sp_train"])
    print(f"P1 Baseline LOSO-RMSE: {result['rmse']:.4f} (avg_iter={result['avg_iter']})")
    return result


if __name__ == "__main__":
    print("Running P1 baseline verification...")
    result = run_p1_baseline()
    print(f"\nResult: LOSO-RMSE = {result['rmse']:.4f}")
    print(f"Per-species RMSE:")
    for sp_id, rmse in sorted(result["sp_rmses"].items()):
        print(f"  Species {sp_id:2d}: {rmse:.2f}")
