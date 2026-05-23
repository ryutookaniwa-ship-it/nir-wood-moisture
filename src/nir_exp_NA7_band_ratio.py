"""
Experiment NA7: 物理バンド比 — アプローチ4
=================================================================
Beer-Lambert則: A = ε * c * l
バンド比 A(水吸収帯)/A(木材構造帯) でパス長差をキャンセル → 種非依存な含水率代理変数

水吸収帯:
  ~5200 cm-1 (5000-5400): 水分子の組み合わせ帯 (最強)
  ~6900 cm-1 (6700-7200): OH第1倍音帯

参照帯(木材構造, 含水率非依存):
  ~5800 cm-1 (5600-6000): CH第1倍音 (セルロース/リグニン)
  ~4400 cm-1 (4200-4600): CH組み合わせ帯

比較:
  A) バンド比 + PLS
  B) バンド比 + LGBM
  C) バンド比 + EPO後スペクトル + LGBM(P1)
  D) 水バンド平均のみ + EPO + LGBM
"""
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import lightgbm as lgb

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import load_data, loso_folds, loso_rmse, save_submission

import warnings; warnings.filterwarnings("ignore")

EXP = "NA7"

data = load_data()
y = data["y_train"]; sp = data["sp_train"]
wns = data["wns"]
X_tr = data["X_train_raw"]; X_te = data["X_test_raw"]

P1_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1, n_jobs=-1,
    random_state=42, learning_rate=0.02, num_leaves=63,
    feature_fraction=0.07, min_child_samples=10,
)

def band_mean(X, wns, lo, hi):
    mask = (wns >= lo) & (wns <= hi)
    return X[:, mask].mean(axis=1)

def band_ratio_features(X, wns):
    w5200 = band_mean(X, wns, 5000, 5400)   # 水組み合わせ帯
    w6900 = band_mean(X, wns, 6700, 7200)   # 水OH第1倍音
    ref58  = band_mean(X, wns, 5600, 6000)  # CH第1倍音(参照)
    ref44  = band_mean(X, wns, 4200, 4600)  # CH組み合わせ(参照)
    eps = 1e-8
    feats = np.column_stack([
        w5200, w6900,
        w5200 / (ref58 + eps),   # 比1: 5200/5800
        w6900 / (ref58 + eps),   # 比2: 6900/5800
        w5200 / (ref44 + eps),   # 比3: 5200/4400
        w6900 / (ref44 + eps),   # 比4: 6900/4400
        (w5200 + w6900) / (ref58 + eps),  # 比5: 合計/5800
        w5200 / (w6900 + eps),   # 比6: 5200/6900 (水帯間)
    ])
    return feats

def compute_epo(X, y, sp, n=5, bw=10.0):
    bins = np.arange(0, y.max() + bw, bw); dirs = []
    for lo in bins[:-1]:
        mask = (y >= lo) & (y < lo + bw)
        if mask.sum() < 4: continue
        sp_u = np.unique(sp[mask])
        if len(sp_u) < 2: continue
        means = np.array([X[mask][sp[mask]==s].mean(0) for s in sp_u])
        inter = means - means.mean(0)
        nc = min(n, inter.shape[0]-1)
        if nc < 1: continue
        pca = PCA(n_components=nc, random_state=42).fit(inter)
        dirs.append(pca.components_)
    if not dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n].T

def apply_epo(X, V): return X - (X @ V) @ V.T

def loso_pls(X, y, sp, nc=5):
    oof = np.zeros(len(y))
    for tr_idx, va_idx, _ in loso_folds(sp):
        n = min(nc, X[tr_idx].shape[1]-1, len(tr_idx)-1)
        pls = PLSRegression(n_components=n).fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = np.clip(pls.predict(X[va_idx]).ravel(), 0, None)
    return loso_rmse(oof, y)

def loso_lgbm(X, label, sp, inv=None):
    oof = np.zeros(len(y)); iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(X[tr_idx], label=label[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=label[va_idx], reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        pred = m.predict(X[va_idx])
        oof[va_idx] = inv(pred) if inv else pred
        iters.append(m.best_iteration)
    rmse = loso_rmse(np.clip(oof, 0, None), y)
    return rmse, int(np.mean(iters)), oof

print(f"=== Experiment {EXP}: 物理バンド比 ===")
print(f"参照: P1 LOSO=15.4725\n")

# バンド比特徴量
F_tr = band_ratio_features(X_tr, wns)
F_te = band_ratio_features(X_te, wns)
feat_names = ["w5200","w6900","5200/5800","6900/5800","5200/4400","6900/4400","sum/5800","5200/6900"]
print("バンド比特徴量 (train相関 vs 含水率):")
for i, nm in enumerate(feat_names):
    c = np.corrcoef(F_tr[:, i], y)[0, 1]
    print(f"  {nm:>12}: corr={c:.4f}")

print(f"\n{'設定':>35}  {'LOSO':>8}")
print("-" * 50)

results = []

# A) バンド比 + PLS
for nc in [3, 5, 8]:
    r = loso_pls(F_tr, y, sp, nc)
    print(f"{'A) 比+PLS':>35}(n={nc})  {r:.4f}")
    results.append((f"A-ratio+PLS(n={nc})", r, F_tr, F_te, y, None))

# B) バンド比 + LGBM
r_b, avg_b, _ = loso_lgbm(F_tr, np.sqrt(y), sp, inv=lambda p: np.clip(p,0,None)**2)
print(f"{'B) 比+LGBM(sqrt)':>35}       {r_b:.4f}")
results.append((f"B-ratio+LGBM", r_b, F_tr, F_te, np.sqrt(y), lambda p: np.clip(p,0,None)**2))

# C) EPO後スペクトル + バンド比 + LGBM
V = compute_epo(X_tr, y, sp, n=5)
X_epo_tr = apply_epo(X_tr, V); X_epo_te = apply_epo(X_te, V)
X_c_tr = np.column_stack([X_epo_tr, F_tr])
X_c_te = np.column_stack([X_epo_te, F_te])
y_p = y ** 0.27
r_c, avg_c, _ = loso_lgbm(X_c_tr, y_p, sp, inv=lambda p: np.clip(p,0,None)**(1/0.27))
print(f"{'C) EPO+比+LGBM(y^0.27)':>35}       {r_c:.4f}")
results.append((f"C-EPO+ratio+LGBM", r_c, X_c_tr, X_c_te, y_p, lambda p: np.clip(p,0,None)**(1/0.27)))

# D) 水バンド平均2列のみ + EPO + LGBM
X_d_tr = np.column_stack([X_epo_tr, F_tr[:, :2]])  # w5200, w6900のみ
X_d_te = np.column_stack([X_epo_te, F_te[:, :2]])
r_d, avg_d, _ = loso_lgbm(X_d_tr, y_p, sp, inv=lambda p: np.clip(p,0,None)**(1/0.27))
print(f"{'D) EPO+水2バンド+LGBM':>35}       {r_d:.4f}")
results.append((f"D-EPO+waterbands+LGBM", r_d, X_d_tr, X_d_te, y_p, lambda p: np.clip(p,0,None)**(1/0.27)))

# E) バンド比のみ Ridge (物理的な直接予測として)
from sklearn.linear_model import Ridge
oof_e = np.zeros(len(y))
for tr_idx, va_idx, _ in loso_folds(sp):
    m = Ridge(alpha=1.0).fit(F_tr[tr_idx], y[tr_idx])
    oof_e[va_idx] = np.clip(m.predict(F_tr[va_idx]), 0, None)
r_e = loso_rmse(oof_e, y)
print(f"{'E) 比+Ridge(物理直接)':>35}       {r_e:.4f}")

print(f"\n{'='*50}")
print(f"サマリ (参照: P1=15.4725)")
print(f"{'='*50}")
for name, rmse, *_ in results:
    diff = rmse - 15.4725
    mark = "UP" if rmse < 15.4725 else "  "
    print(f"  {mark} {name}: LOSO={rmse:.4f} (diff={diff:+.4f})")
print(f"     E) バンド比+Ridge: LOSO={r_e:.4f}")

best = min(results, key=lambda x: x[1])
bname, brmse, bXtr, bXte, blabel, binv = best
print(f"\nベスト: {bname}  LOSO={brmse:.4f}")

# submission生成 (C か D が勝った場合のみ)
if brmse < 15.4725:
    iters = []
    for tr_idx, va_idx, _ in loso_folds(sp):
        dtrain = lgb.Dataset(bXtr[tr_idx], label=blabel[tr_idx])
        dval   = lgb.Dataset(bXtr[va_idx], label=blabel[va_idx], reference=dtrain)
        m = lgb.train(P1_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
        iters.append(m.best_iteration)
    avg_iter = int(np.mean(iters))
    final = lgb.train(P1_PARAMS, lgb.Dataset(bXtr, label=blabel), num_boost_round=avg_iter)
    preds = np.clip(binv(final.predict(bXte)) if binv else final.predict(bXte), 0, None)
    out = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\na7_best.csv"
    save_submission(data["test_ids"], preds, out)
    print(f"avg_iter={avg_iter}")
else:
    print("P1を上回らなかったため提出ファイル未生成")
