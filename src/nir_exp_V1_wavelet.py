"""
Experiment V1: ウェーブレット変換特徴量 + P1パイプライン
==========================================================
ウェーブレット変換はマルチ解像度で広域ベースライン＋狭域吸収ピークを同時に捉える。
IFFTより物理的に適切。

試すバリアント:
  V1a: EPO + db4  level=4 全係数
  V1b: EPO + sym4 level=4 全係数
  V1c: EPO + db4  level=4 近似係数のみ (cA4, 低次元)
  V1d: EPO + db4  level=4 詳細係数のみ (cD1〜cD4, 高周波成分)

ベース: P1 LOSO=15.4725, LB=15.395
"""
import sys
import numpy as np
import pywt
from sklearn.decomposition import PCA
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "V1"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"
P1_LOSO = 15.4725


def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width; mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species: continue
        sp_means = np.array([X[mask][sp[mask] == s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0] - 1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T


def apply_epo(X, V): return X - (X @ V) @ V.T


def wavelet_features(X, wavelet='db4', level=4, mode='all'):
    """
    各サンプルスペクトルにDWTを適用し係数を返す。
    mode='all'   : 近似+全詳細係数を結合
    mode='approx': 近似係数のみ
    mode='detail': 詳細係数のみ (cD1〜cDlevel)
    """
    feats = []
    for row in X:
        coeffs = pywt.wavedec(row, wavelet=wavelet, level=level)
        # coeffs = [cA_level, cD_level, ..., cD1]
        if mode == 'all':
            feat = np.concatenate(coeffs)
        elif mode == 'approx':
            feat = coeffs[0]
        elif mode == 'detail':
            feat = np.concatenate(coeffs[1:])
        feats.append(feat)
    return np.array(feats)


# ── データ準備 ─────────────────────────────────────────────────────────────────
data = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

ref     = X_train_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg  = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V       = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

print("ウェーブレット係数を計算中...")
Wtr_db4_all    = wavelet_features(Xtr_epo, 'db4',  level=4, mode='all')
Wte_db4_all    = wavelet_features(Xte_epo, 'db4',  level=4, mode='all')
Wtr_sym4_all   = wavelet_features(Xtr_epo, 'sym4', level=4, mode='all')
Wte_sym4_all   = wavelet_features(Xte_epo, 'sym4', level=4, mode='all')
Wtr_db4_approx = wavelet_features(Xtr_epo, 'db4',  level=4, mode='approx')
Wte_db4_approx = wavelet_features(Xte_epo, 'db4',  level=4, mode='approx')
Wtr_db4_detail = wavelet_features(Xtr_epo, 'db4',  level=4, mode='detail')
Wte_db4_detail = wavelet_features(Xte_epo, 'db4',  level=4, mode='detail')
print(f"  db4 all:    {Wtr_db4_all.shape[1]}次元")
print(f"  db4 approx: {Wtr_db4_approx.shape[1]}次元")
print(f"  db4 detail: {Wtr_db4_detail.shape[1]}次元")

variants = {
    "V1a": (np.hstack([Xtr_epo, Wtr_db4_all]),
            np.hstack([Xte_epo, Wte_db4_all]),
            f"EPO+db4全係数  {1555+Wtr_db4_all.shape[1]}次元"),
    "V1b": (np.hstack([Xtr_epo, Wtr_sym4_all]),
            np.hstack([Xte_epo, Wte_sym4_all]),
            f"EPO+sym4全係数 {1555+Wtr_sym4_all.shape[1]}次元"),
    "V1c": (np.hstack([Xtr_epo, Wtr_db4_approx]),
            np.hstack([Xte_epo, Wte_db4_approx]),
            f"EPO+db4近似のみ {1555+Wtr_db4_approx.shape[1]}次元"),
    "V1d": (np.hstack([Xtr_epo, Wtr_db4_detail]),
            np.hstack([Xte_epo, Wte_db4_detail]),
            f"EPO+db4詳細のみ {1555+Wtr_db4_detail.shape[1]}次元"),
}

params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

p = 0.27
y_trans = y_train ** p
inv = lambda pred: np.clip(pred, 0, None) ** (1.0 / p)

print(f"\n=== Experiment {EXP}: ウェーブレット特徴量 ===")
print(f"ベース P1: LOSO={P1_LOSO}, LB=15.395\n")
print(f"{'variant':<6}  {'LOSO':>8}  {'avg_iter':>9}  {'vs P1':>7}  説明")
print("-" * 78)

best_rmse = P1_LOSO
best_key  = None
best_data = None

for key, (Xtr, Xte, desc) in variants.items():
    oof_trans = np.zeros(len(y_trans))
    iters = []
    for tr_idx, va_idx, _ in loso_folds(sp_train):
        dtrain = lgb.Dataset(Xtr[tr_idx], label=y_trans[tr_idx])
        dval   = lgb.Dataset(Xtr[va_idx], label=y_trans[va_idx], reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof_trans[va_idx] = m.predict(Xtr[va_idx])
        iters.append(m.best_iteration)
    oof  = inv(oof_trans)
    rmse = loso_rmse(oof, y_train)
    avg_r = int(np.mean(iters))
    diff = rmse - P1_LOSO
    flag = " <-- best" if rmse < P1_LOSO else ""
    print(f"  {key:<6}  {rmse:8.4f}  {avg_r:9d}  {diff:+7.4f}  {desc}{flag}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_key  = key
        best_data = (Xtr, Xte, avg_r)

print()
if best_key:
    print(f"Best: {best_key}  LOSO={best_rmse:.4f}  vs P1: {best_rmse - P1_LOSO:+.4f}")
    Xtr_b, Xte_b, avg_r_b = best_data
    dtrain_f = lgb.Dataset(Xtr_b, label=y_train ** p)
    final = lgb.train(params, dtrain_f,
                      num_boost_round=avg_r_b,
                      callbacks=[lgb.log_evaluation(-1)])
    preds = inv(final.predict(Xte_b))
    OUT = f"{OUT_DIR}/submission_{best_key}_wavelet.csv"
    save_submission(test_ids, preds, OUT)
    submit_to_signate(OUT, f"{best_key}: wavelet LOSO={best_rmse:.4f}", loso=best_rmse)
else:
    print(f"全バリアントがP1(LOSO={P1_LOSO})を超えず → 提出なし")
