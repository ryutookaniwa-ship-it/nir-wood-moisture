"""
Experiment NA4: fold内CARS波長選択 + PLS (honest LOSO)
=================================================================
NA2-BのCARS波長選択はfold外(全訓練種)で実施 → LOSO楽観的(13.93 → LB=33.22)

今回: 各foldで12種のみを使ってCARS選択 → 本当の未知種汎化性能を評価
(ただし最終テスト予測は全13種CARS → PLS)

比較:
  A) fold内CARS + PLS (EPOなし)
  B) fold内CARS + PLS (EPOあり, fold内でEPO計算)
  C) fold内CARS + PLS (EPO全体で計算, CARS fold内)

注: fold内EPOはL4実験で悪化(+5.18)が確認済み → Cが主目的
"""
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import load_data, msc, sg_deriv, loso_folds, loso_rmse, save_submission

import warnings; warnings.filterwarnings("ignore")

EXP = "NA4"

# ── データ・前処理 ────────────────────────────────────────────────────────────
data = load_data()
y = data["y_train"]
sp = data["sp_train"]
wns = data["wns"]

ref = data["X_train_raw"].mean(axis=0)
X_sg_tr = sg_deriv(msc(data["X_train_raw"], ref), window=9, polyorder=2)
X_sg_te = sg_deriv(msc(data["X_test_raw"],  ref), window=9, polyorder=2)


def compute_epo(X, y, sp, n=5, bw=10.0):
    bins = np.arange(0, y.max() + bw, bw)
    dirs = []
    for lo in bins[:-1]:
        mask = (y >= lo) & (y < lo + bw)
        if mask.sum() < 4: continue
        sp_u = np.unique(sp[mask])
        if len(sp_u) < 2: continue
        means = np.array([X[mask][sp[mask] == s].mean(0) for s in sp_u])
        inter = means - means.mean(0)
        nc = min(n, inter.shape[0] - 1)
        if nc < 1: continue
        pca = PCA(n_components=nc, random_state=42).fit(inter)
        dirs.append(pca.components_)
    if not dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(dirs)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n].T

def apply_epo(X, V): return X - (X @ V) @ V.T


def cars_select_mask(X, y, n_iter=50, n_comp=10, min_waves=5):
    """X上でCARS波長選択を実行。選択マスクを返す。"""
    n_samples, n_features = X.shape
    current_mask = np.ones(n_features, dtype=bool)
    wave_counts = np.round(
        np.exp(np.linspace(np.log(n_features), np.log(min_waves), n_iter))
    ).astype(int)
    wave_counts = np.clip(wave_counts, min_waves, n_features)

    history = []
    for n_keep in wave_counts:
        X_sel = X[:, current_mask]
        nc = min(n_comp, X_sel.shape[1] - 1, n_samples - 1)
        if nc < 1: break
        pls = PLSRegression(n_components=nc)
        pls.fit(X_sel, y)
        coefs = np.abs(pls.coef_.ravel())
        sel_local = np.argsort(coefs)[::-1][:n_keep]
        orig = np.where(current_mask)[0]
        new_mask = np.zeros(n_features, dtype=bool)
        new_mask[orig[sel_local]] = True
        current_mask = new_mask
        history.append((n_keep, current_mask.copy()))

    # 最終マスク(最小波長数)を返す — fold内ではCV不要
    return current_mask


def loso_honest_cars_pls(X, y, sp, use_epo_global=False, X_epo=None):
    """
    fold内CARS + PLS。
    use_epo_global=True: fold分割後のX_epoを使いCARSはfold内。
    """
    oof = np.zeros(len(y))
    n_waves_list = []

    for tr_idx, va_idx, sp_val in loso_folds(sp):
        X_use = X_epo if use_epo_global else X

        # fold内のみでCARS
        X_tr_fold = X_use[tr_idx]
        y_tr_fold = y[tr_idx]

        mask = cars_select_mask(X_tr_fold, y_tr_fold, n_iter=40, n_comp=10, min_waves=5)
        n_waves_list.append(mask.sum())

        X_sel_tr = X_tr_fold[:, mask]
        X_sel_va = X_use[va_idx][:, mask]

        nc = min(10, mask.sum() - 1, len(tr_idx) - 1)
        if nc < 1:
            oof[va_idx] = y.mean()
            continue

        pls = PLSRegression(n_components=nc)
        pls.fit(X_sel_tr, y_tr_fold)
        oof[va_idx] = np.clip(pls.predict(X_sel_va).ravel(), 0, None)

    rmse = loso_rmse(oof, y)
    print(f"  fold内波長数: min={min(n_waves_list)}, max={max(n_waves_list)}, mean={np.mean(n_waves_list):.1f}")
    return rmse, oof


print(f"=== Experiment {EXP}: fold内CARS波長選択 + PLS (honest LOSO) ===")
print(f"参照: P1=15.4725, NA2-B(fold外CARS)=13.93→LB=33.22\n")

# ── A) fold内CARS + PLS (EPOなし) ─────────────────────────────────────────
print("A) fold内CARS + PLS (EPOなし)")
rmse_a, oof_a = loso_honest_cars_pls(X_sg_tr, y, sp, use_epo_global=False)
print(f"  → LOSO={rmse_a:.4f}\n")

# ── C) EPO全体計算 + fold内CARS + PLS (メイン) ────────────────────────────
print("C) EPO(全体) + fold内CARS + PLS")
V_full = compute_epo(X_sg_tr, y, sp, n=5)
X_epo_tr = apply_epo(X_sg_tr, V_full)
X_epo_te = apply_epo(X_sg_te, V_full)
rmse_c, oof_c = loso_honest_cars_pls(X_epo_tr, y, sp, use_epo_global=True, X_epo=X_epo_tr)
print(f"  → LOSO={rmse_c:.4f}\n")

# ── B) fold内EPO + fold内CARS + PLS ──────────────────────────────────────
print("B) fold内EPO + fold内CARS + PLS")
oof_b = np.zeros(len(y))
n_waves_b = []
for tr_idx, va_idx, sp_val in loso_folds(sp):
    X_tr_f = X_sg_tr[tr_idx]; y_tr_f = y[tr_idx]; sp_tr_f = sp[tr_idx]
    V_f = compute_epo(X_tr_f, y_tr_f, sp_tr_f, n=5)
    X_tr_epo_f = apply_epo(X_tr_f, V_f)
    X_va_epo_f = apply_epo(X_sg_tr[va_idx], V_f)

    mask = cars_select_mask(X_tr_epo_f, y_tr_f, n_iter=40, n_comp=10, min_waves=5)
    n_waves_b.append(mask.sum())
    nc = min(10, mask.sum() - 1, len(tr_idx) - 1)
    if nc < 1:
        oof_b[va_idx] = y.mean()
        continue
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr_epo_f[:, mask], y_tr_f)
    oof_b[va_idx] = np.clip(pls.predict(X_va_epo_f[:, mask]).ravel(), 0, None)

rmse_b = loso_rmse(oof_b, y)
print(f"  fold内波長数: min={min(n_waves_b)}, max={max(n_waves_b)}, mean={np.mean(n_waves_b):.1f}")
print(f"  → LOSO={rmse_b:.4f}\n")

# ── サマリ ────────────────────────────────────────────────────────────────
print("=" * 55)
print("サマリ")
print("=" * 55)
print(f"  参照 P1 (EPO+LGBM)         : LOSO=15.4725")
print(f"  参照 NA2-B (fold外CARS+PLS): LOSO=13.93 → LB=33.22")
print(f"  A) fold内CARS+PLS (EPOなし): LOSO={rmse_a:.4f}")
print(f"  B) fold内EPO+CARS+PLS      : LOSO={rmse_b:.4f}")
print(f"  C) EPO全体+fold内CARS+PLS  : LOSO={rmse_c:.4f}")

best_rmse = min(rmse_a, rmse_b, rmse_c)
best_label = ["A","B","C"][[rmse_a, rmse_b, rmse_c].index(best_rmse)]
print(f"\nベスト: {best_label}  LOSO={best_rmse:.4f}")

# 最良でsubmission生成 (Cが最良の場合想定)
if best_label == "C":
    mask_final = cars_select_mask(X_epo_tr, y, n_iter=40, n_comp=10, min_waves=5)
    X_sel_tr = X_epo_tr[:, mask_final]
    X_sel_te = X_epo_te[:, mask_final]
elif best_label == "A":
    mask_final = cars_select_mask(X_sg_tr, y, n_iter=40, n_comp=10, min_waves=5)
    X_sel_tr = X_sg_tr[:, mask_final]
    X_sel_te = X_sg_te[:, mask_final]
else:
    V_b_full = compute_epo(X_sg_tr, y, sp, n=5)
    X_b_tr = apply_epo(X_sg_tr, V_b_full)
    X_b_te = apply_epo(X_sg_te, V_b_full)
    mask_final = cars_select_mask(X_b_tr, y, n_iter=40, n_comp=10, min_waves=5)
    X_sel_tr = X_b_tr[:, mask_final]
    X_sel_te = X_b_te[:, mask_final]

nc_final = min(10, mask_final.sum() - 1)
pls_final = PLSRegression(n_components=nc_final)
pls_final.fit(X_sel_tr, y)
preds = np.clip(pls_final.predict(X_sel_te).ravel(), 0, None)
out_path = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na4_best.csv"
save_submission(data["test_ids"], preds, out_path)
print(f"n_waves={mask_final.sum()}, n_comp={nc_final}")
