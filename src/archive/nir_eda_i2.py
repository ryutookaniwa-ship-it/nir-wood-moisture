"""
EDA: I2パイプライン スコア改善ヒント発掘
=========================================
現LBベスト: I2 (MSC+SG(w=9,p=2)+EPO(n=5)+LGBM) = LB 16.101, LOSO 15.73

4つの問いに答える:
  Q1. EPO後の空間でテスト樹種はどこにいるか (PCA可視化)
  Q2. 誤差が大きいサンプルの共通パターン (MC帯・樹種・スペクトル偏差)
  Q3. sp15はまだボトルネックか (樹種別RMSE)
  Q4. EPOが除去した方向に水分シグナルはあったか (EPO前後の相関比較)

出力: output/nir-wood-moisture/eda_i2/ 以下にPNG保存
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse, LGBM_BASE_PARAMS
)

OUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture\eda_i2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── パイプライン ──────────────────────────────────────────────────────────────
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

print("データ読み込み中...")
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; sp_train = data["sp_train"]; wns = data["wns"]
y_sqrt = np.sqrt(y_train)

ref = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr = apply_epo(Xtr_sg, V)
Xte = apply_epo(Xte_sg, V)

print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
print(f"Train species: {sorted(set(sp_train))}")
print(f"y_train range: {y_train.min():.1f} ~ {y_train.max():.1f}%\n")

# ── LOSO OOF取得 ──────────────────────────────────────────────────────────────
print("LOSO OOF計算中...")
params = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

oof = np.zeros(len(y_train)); best_iters = []
for tr_idx, va_idx, sp in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(Xtr[va_idx], label=y_sqrt[va_idx], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof[va_idx] = np.clip(m.predict(Xtr[va_idx]), 0, None) ** 2
    best_iters.append(m.best_iteration)

loso = loso_rmse(oof, y_train)
print(f"LOSO-RMSE: {loso:.4f}  avg_iter: {int(np.mean(best_iters))}\n")

residuals = oof - y_train

# ── 全データでfinalモデル学習（特徴重要度用） ─────────────────────────────────
dtrain_f = lgb.Dataset(Xtr, label=y_sqrt)
final_model = lgb.train(params, dtrain_f, num_boost_round=int(np.mean(best_iters)),
                        callbacks=[lgb.log_evaluation(-1)])

# =============================================================================
# Q1: EPO後の空間でテスト樹種はどこにいるか
# =============================================================================
print("Q1: EPO後PCA可視化...")
all_X = np.vstack([Xtr, Xte])
pca2 = PCA(n_components=2, random_state=42)
all_pca = pca2.fit_transform(all_X)
tr_pca = all_pca[:len(Xtr)]
te_pca = all_pca[len(Xtr):]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左: 訓練樹種ごとに色分け
sp_list = sorted(set(sp_train))
cmap = cm.get_cmap("tab20", len(sp_list))
for i, sp in enumerate(sp_list):
    idx = sp_train == sp
    axes[0].scatter(tr_pca[idx, 0], tr_pca[idx, 1],
                    color=cmap(i), label=f"sp{sp}", alpha=0.6, s=15)
axes[0].scatter(te_pca[:, 0], te_pca[:, 1],
                color="black", marker="x", s=40, linewidths=1.5, label="TEST", zorder=5)
axes[0].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("Q1: EPO後 PCA — 訓練樹種 vs テスト")
axes[0].legend(fontsize=7, ncol=2, loc="upper right")

# 右: MC値で色分け（訓練）+ テスト位置
sc = axes[1].scatter(tr_pca[:, 0], tr_pca[:, 1],
                     c=y_train, cmap="viridis", alpha=0.5, s=15,
                     vmin=0, vmax=np.percentile(y_train, 95))
plt.colorbar(sc, ax=axes[1], label="MC (%)")
axes[1].scatter(te_pca[:, 0], te_pca[:, 1],
                color="red", marker="x", s=40, linewidths=1.5, label="TEST", zorder=5)
axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_title("Q1: EPO後 PCA — MC値 vs テスト位置")
axes[1].legend()

fig.suptitle(f"I2パイプライン EPO後空間 (LOSO={loso:.4f})", fontsize=12)
fig.tight_layout()
path = str(OUT_DIR / "q1_epo_pca.png")
fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"  → {path}")

# テスト樹種の最近傍訓練樹種を特定
print("\n  テスト各サンプルの最近傍訓練樹種 (EPO空間のユークリッド距離):")
sp_means_pca = {sp: tr_pca[sp_train == sp].mean(axis=0) for sp in sp_list}
for i_te in range(len(te_pca)):
    dists = {sp: np.linalg.norm(te_pca[i_te] - mu) for sp, mu in sp_means_pca.items()}
    top3 = sorted(dists, key=dists.get)[:3]
    if i_te < 5:
        print(f"    test[{i_te}]: 近傍 = sp{top3[0]}(d={dists[top3[0]]:.3f}), "
              f"sp{top3[1]}(d={dists[top3[1]]:.3f}), sp{top3[2]}(d={dists[top3[2]]:.3f})")

# =============================================================================
# Q2: 誤差が大きいサンプルの共通パターン
# =============================================================================
print("\nQ2: 誤差パターン分析...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (0,0) 予測 vs 実測
axes[0,0].scatter(y_train, oof, alpha=0.3, s=10, c=sp_train, cmap="tab20")
lim = max(y_train.max(), oof.max()) * 1.05
axes[0,0].plot([0, lim], [0, lim], "r--", lw=1)
axes[0,0].set_xlabel("実測 MC (%)"); axes[0,0].set_ylabel("予測 MC (%)")
axes[0,0].set_title("OOF: 予測 vs 実測")

# (0,1) MC帯別RMSE
mc_bins = [0, 10, 20, 30, 50, 80, 120, 200, 400]
bin_labels = [f"{mc_bins[i]}-{mc_bins[i+1]}" for i in range(len(mc_bins)-1)]
bin_rmse, bin_n = [], []
for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
    mask = (y_train >= lo) & (y_train < hi)
    if mask.sum() > 0:
        bin_rmse.append(np.sqrt(np.mean((residuals[mask])**2)))
        bin_n.append(mask.sum())
    else:
        bin_rmse.append(0); bin_n.append(0)

bars = axes[0,1].bar(bin_labels, bin_rmse, color="steelblue", edgecolor="white")
ax2 = axes[0,1].twinx()
ax2.plot(bin_labels, bin_n, "o--", color="orange", label="n")
ax2.set_ylabel("サンプル数", color="orange")
axes[0,1].set_xlabel("MC帯 (%)"); axes[0,1].set_ylabel("RMSE")
axes[0,1].set_title("MC帯別RMSE"); axes[0,1].tick_params(axis="x", rotation=30)

# (1,0) 残差 vs MC
sc2 = axes[1,0].scatter(y_train, residuals, alpha=0.3, s=10, c=sp_train, cmap="tab20")
axes[1,0].axhline(0, color="red", lw=1)
axes[1,0].axvline(30, color="gray", lw=1, ls="--", label="FSP~30%")
axes[1,0].set_xlabel("実測 MC (%)"); axes[1,0].set_ylabel("残差 (pred - actual)")
axes[1,0].set_title("残差 vs MC (色=樹種)")
axes[1,0].legend(fontsize=8)

# (1,1) |残差|上位30サンプルのMC分布
top_idx = np.argsort(np.abs(residuals))[-30:][::-1]
axes[1,1].bar(range(30), np.abs(residuals[top_idx]), color="coral")
axes[1,1].set_xticks(range(30))
axes[1,1].set_xticklabels([f"sp{sp_train[i]}" for i in top_idx], rotation=90, fontsize=7)
axes[1,1].set_xlabel("サンプル (樹種番号)"); axes[1,1].set_ylabel("|残差|")
axes[1,1].set_title("|残差|上位30サンプル")

fig.suptitle("Q2: 誤差パターン分析 (I2 OOF)", fontsize=12)
fig.tight_layout()
path = str(OUT_DIR / "q2_error_patterns.png")
fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"  → {path}")

# =============================================================================
# Q3: sp15はまだボトルネックか (樹種別RMSE)
# =============================================================================
print("\nQ3: 樹種別RMSE...")
sp_stats = []
for sp in sp_list:
    idx = sp_train == sp
    sp_rmse = np.sqrt(np.mean((residuals[idx])**2))
    sp_bias = residuals[idx].mean()
    sp_stats.append({"sp": sp, "n": idx.sum(), "rmse": sp_rmse, "bias": sp_bias,
                     "mc_mean": y_train[idx].mean(), "mc_max": y_train[idx].max()})

df_sp = pd.DataFrame(sp_stats).sort_values("rmse", ascending=False)
print(df_sp.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ["red" if r > loso * 1.5 else "steelblue" for r in df_sp["rmse"]]
axes[0].bar(df_sp["sp"].astype(str), df_sp["rmse"], color=colors)
axes[0].axhline(loso, color="black", lw=1.5, ls="--", label=f"全体 LOSO={loso:.2f}")
axes[0].set_xlabel("樹種番号"); axes[0].set_ylabel("RMSE")
axes[0].set_title("Q3: 樹種別RMSE (赤=全体×1.5以上)")
axes[0].legend()

axes[1].scatter(df_sp["mc_max"], df_sp["rmse"], s=df_sp["n"]*3, alpha=0.7, color="steelblue")
for _, row in df_sp.iterrows():
    axes[1].annotate(f"sp{int(row['sp'])}", (row["mc_max"], row["rmse"]),
                     fontsize=8, ha="left", va="bottom")
axes[1].set_xlabel("樹種内 MC最大値 (%)"); axes[1].set_ylabel("RMSE")
axes[1].set_title("MC最大値 vs RMSE (丸の大きさ=サンプル数)")

fig.suptitle("Q3: 樹種別RMSE詳細 (I2 OOF)", fontsize=12)
fig.tight_layout()
path = str(OUT_DIR / "q3_species_rmse.png")
fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"  → {path}")

# =============================================================================
# Q4: EPOが除去した方向に水分シグナルはあったか
# =============================================================================
print("\nQ4: EPO効果の検証...")
# EPO前後の各サンプルとMCの相関
corr_before = np.array([np.corrcoef(Xtr_sg[:, i], y_train)[0,1] for i in range(Xtr_sg.shape[1])])
corr_after  = np.array([np.corrcoef(Xtr[:, i],    y_train)[0,1] for i in range(Xtr.shape[1])])

# EPO方向V のMCとの相関
epo_proj_before = Xtr_sg @ V  # (n, 5)
epo_proj_after  = Xtr @ V     # (n, 5) ≈ 0 after EPO

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (0,0) EPO前後の波数別MC相関
axes[0,0].plot(wns, corr_before, lw=0.8, alpha=0.7, label="EPO前", color="gray")
axes[0,0].plot(wns, corr_after,  lw=0.8, alpha=0.9, label="EPO後", color="steelblue")
axes[0,0].axhline(0, color="black", lw=0.5)
for wn in [5187, 6896, 8333]:
    if wns.min() <= wn <= wns.max():
        axes[0,0].axvline(wn, color="red", lw=0.8, ls="--", alpha=0.6)
axes[0,0].set_xlabel("波数 (cm⁻¹)"); axes[0,0].set_ylabel("Pearson r (波数 vs MC)")
axes[0,0].set_title("Q4: EPO前後の波数別MC相関")
axes[0,0].legend(); axes[0,0].invert_xaxis()

# (0,1) EPO5方向の波数プロファイル (除去された方向の可視化)
for i in range(V.shape[1]):
    axes[0,1].plot(wns, V[:, i], lw=0.8, alpha=0.8, label=f"EPO dir {i+1}")
axes[0,1].set_xlabel("波数 (cm⁻¹)"); axes[0,1].set_ylabel("成分値")
axes[0,1].set_title("EPOが除去した5方向 (波数プロファイル)")
axes[0,1].legend(fontsize=8); axes[0,1].invert_xaxis()

# (1,0) EPO前後の|相関|の差分 (正 = EPOで相関が下がった = 水分情報が失われた)
diff_corr = np.abs(corr_before) - np.abs(corr_after)
axes[1,0].fill_between(wns, diff_corr, 0,
                        where=diff_corr > 0, color="red",  alpha=0.5, label="|r|減少 (EPOで除去)")
axes[1,0].fill_between(wns, diff_corr, 0,
                        where=diff_corr < 0, color="blue", alpha=0.5, label="|r|増加 (EPOで強調)")
axes[1,0].axhline(0, color="black", lw=0.5)
for wn in [5187, 6896, 8333]:
    if wns.min() <= wn <= wns.max():
        axes[1,0].axvline(wn, color="red", lw=0.8, ls="--", alpha=0.6)
axes[1,0].set_xlabel("波数 (cm⁻¹)"); axes[1,0].set_ylabel("Δ|r| (EPO前-EPO後)")
axes[1,0].set_title("Q4: EPOによる水分相関の変化量")
axes[1,0].legend(); axes[1,0].invert_xaxis()

# (1,1) 特徴量重要度 (EPO後)
imp = final_model.feature_importance(importance_type="gain")
feat_names = final_model.feature_name()
indices = [int(f.split("_")[1]) for f in feat_names]
imp_by_wn = np.zeros(len(wns))
for idx_f, imp_v in zip(indices, imp):
    if idx_f < len(wns):
        imp_by_wn[idx_f] = imp_v
imp_by_wn /= (imp_by_wn.sum() + 1e-9)

axes[1,1].fill_between(wns, imp_by_wn, alpha=0.7, color="green")
for wn in [5187, 6896, 8333]:
    if wns.min() <= wn <= wns.max():
        axes[1,1].axvline(wn, color="red", lw=0.8, ls="--", alpha=0.8,
                          label=f"{wn} cm⁻¹")
axes[1,1].set_xlabel("波数 (cm⁻¹)"); axes[1,1].set_ylabel("重要度 (gain, 正規化)")
axes[1,1].set_title("LGBM 特徴量重要度 (EPO後)")
axes[1,1].legend(fontsize=8); axes[1,1].invert_xaxis()

fig.suptitle("Q4: EPO効果の検証 (I2)", fontsize=12)
fig.tight_layout()
path = str(OUT_DIR / "q4_epo_effect.png")
fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"  → {path}")

# =============================================================================
# サマリー出力
# =============================================================================
print("\n" + "="*60)
print("EDA サマリー (I2: LOSO={:.4f})".format(loso))
print("="*60)

print("\n【Q3: 樹種別RMSE Top5】")
for _, row in df_sp.head(5).iterrows():
    print(f"  sp{int(row['sp'])}: RMSE={row['rmse']:.2f}, bias={row['bias']:+.2f}, "
          f"n={int(row['n'])}, MC_max={row['mc_max']:.1f}%")

print("\n【Q4: EPOによる水分相関の変化】")
top_lost  = np.argsort(diff_corr)[-5:][::-1]
top_gain  = np.argsort(diff_corr)[:5]
print("  EPOで最も水分相関が失われた波数:")
for i in top_lost:
    print(f"    {wns[i]:.0f} cm-1: delta|r|={diff_corr[i]:+.4f} "
          f"(before={abs(corr_before[i]):.4f}, after={abs(corr_after[i]):.4f})")
print("  EPOで水分相関が増加した波数:")
for i in top_gain:
    print(f"    {wns[i]:.0f} cm-1: delta|r|={diff_corr[i]:+.4f} "
          f"(before={abs(corr_before[i]):.4f}, after={abs(corr_after[i]):.4f})")

print(f"\n図の保存先: {OUT_DIR}")
print("  q1_epo_pca.png     - テスト樹種の位置")
print("  q2_error_patterns.png - MC帯別・残差パターン")
print("  q3_species_rmse.png   - 樹種別RMSE")
print("  q4_epo_effect.png     - EPO効果")
