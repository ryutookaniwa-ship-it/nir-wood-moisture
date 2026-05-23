"""
EDA: P1ベースの深掘り分析
=========================
目的: W1〜W3失敗後、新たな打ち手を見つけるためのEDA

分析項目:
  1. P1 OOF残差のMCレンジ別・樹種別詳細
  2. EPO投影空間: train13種 vs test6種の分布 (PCA可視化)
  3. P1 特徴量重要度スペクトル (どの波数が重要か)
  4. テスト予測値の分布 vs 訓練樹種のMC分布
  5. sp15の特異性分析 (スペクトル vs 他種)
  6. 残差の構造: MC・樹種・スペクトル強度との相関
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import load_data, msc, sg_deriv, loso_folds, loso_rmse, LGBM_BASE_PARAMS

OUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\eda_p1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 定数 ─────────────────────────────────────────────────────────────────────
P1_LOSO    = 15.4725
P_TRANS    = 0.27
EPO_N      = 5
SG_W, SG_P = 9, 2
PARAMS = {**LGBM_BASE_PARAMS,
          "learning_rate": 0.02, "num_leaves": 63,
          "feature_fraction": 0.07, "min_child_samples": 10}

# ── データ・前処理 ─────────────────────────────────────────────────────────────
data     = load_data()
y        = data["y_train"]
Xtr_raw  = data["X_train_raw"]
Xte_raw  = data["X_test_raw"]
sp_train = data["sp_train"]
test_ids = data["test_ids"]
wns      = data["wns"]

ref     = Xtr_raw.mean(axis=0)
Xtr_sg  = sg_deriv(msc(Xtr_raw, ref), window=SG_W, polyorder=SG_P)
Xte_sg  = sg_deriv(msc(Xte_raw, ref), window=SG_W, polyorder=SG_P)

# ── EPO ──────────────────────────────────────────────────────────────────────
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

V       = compute_epo_matrix(Xtr_sg, y, sp_train, n_components=EPO_N)
Xtr_epo = apply_epo(Xtr_sg, V)
Xte_epo = apply_epo(Xte_sg, V)

# ── P1 LOSO-CV: OOF + feature importance 収集 ─────────────────────────────
print("Running P1 LOSO-CV...")
y_trans   = y ** P_TRANS
oof_trans = np.zeros(len(y))
imp_accum = np.zeros(Xtr_epo.shape[1])
iters     = []

for tr_idx, va_idx, sp_id in loso_folds(sp_train):
    dtrain = lgb.Dataset(Xtr_epo[tr_idx], label=y_trans[tr_idx])
    dval   = lgb.Dataset(Xtr_epo[va_idx], label=y_trans[va_idx], reference=dtrain)
    m = lgb.train(PARAMS, dtrain, num_boost_round=3000,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof_trans[va_idx] = m.predict(Xtr_epo[va_idx])
    iters.append(m.best_iteration)
    imp = m.feature_importance(importance_type="gain")
    for fname, iv in zip(m.feature_name(), imp):
        idx = int(fname.split("_")[1])
        if idx < len(imp_accum):
            imp_accum[idx] += iv

oof      = np.clip(oof_trans, 0, None) ** (1.0 / P_TRANS)
rmse     = loso_rmse(oof, y)
avg_iter = int(np.mean(iters))
print(f"P1 LOSO-RMSE = {rmse:.4f}  avg_iter={avg_iter}")

# 全データでfinalモデルを訓練してtest予測
dtrain_f = lgb.Dataset(Xtr_epo, label=y_trans)
final_m  = lgb.train(PARAMS, dtrain_f, num_boost_round=avg_iter,
                     callbacks=[lgb.log_evaluation(-1)])
test_preds = np.clip(final_m.predict(Xte_epo), 0, None) ** (1.0 / P_TRANS)
print(f"Test preds: min={test_preds.min():.1f}, max={test_preds.max():.1f}, mean={test_preds.mean():.1f}")

residuals = oof - y
imp_norm  = imp_accum / (imp_accum.sum() + 1e-9)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 1: MCレンジ別誤差の詳細
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 1] Error by MC range...")
bins_mc  = [0, 15, 30, 50, 80, 120, 160, np.inf]
labels_mc = ["0-15", "15-30", "30-50", "50-80", "80-120", "120-160", "160+"]
bin_idx   = np.digitize(y, bins_mc) - 1

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

rmse_by_bin = []; bias_by_bin = []; n_by_bin = []
for b in range(len(labels_mc)):
    mask = bin_idx == b
    if mask.sum() > 0:
        rmse_by_bin.append(np.sqrt(np.mean((y[mask] - oof[mask])**2)))
        bias_by_bin.append(np.mean(oof[mask] - y[mask]))
        n_by_bin.append(mask.sum())
    else:
        rmse_by_bin.append(0); bias_by_bin.append(0); n_by_bin.append(0)

x = np.arange(len(labels_mc))
bars = axes[0].bar(x, rmse_by_bin, color=["salmon" if r > rmse else "steelblue" for r in rmse_by_bin])
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_mc)
axes[0].axhline(rmse, color="red", ls="--", lw=1.5, label=f"Overall={rmse:.2f}")
axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE by MC range (%)"); axes[0].legend()
for bar, n in zip(bars, n_by_bin):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"n={n}", ha="center", fontsize=7)

bars2 = axes[1].bar(x, bias_by_bin, color=["crimson" if b > 0 else "steelblue" for b in bias_by_bin])
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_mc)
axes[1].axhline(0, color="black", lw=1)
axes[1].set_ylabel("Bias (Pred-Actual)"); axes[1].set_title("Bias by MC range")

# Predicted vs Actual scatter with MC coloring
sc = axes[2].scatter(y, oof, c=y, cmap="plasma", alpha=0.5, s=12, vmin=0, vmax=200)
axes[2].plot([0, y.max()], [0, y.max()], "r--", lw=1)
axes[2].axvline(30, color="orange", lw=1, ls="--", alpha=0.7, label="FSP=30%")
axes[2].set_xlabel("Actual MC (%)"); axes[2].set_ylabel("Predicted MC (%)")
axes[2].set_title("OOF: Predicted vs Actual")
plt.colorbar(sc, ax=axes[2], label="Actual MC (%)")
axes[2].legend(fontsize=8)

fig.suptitle("Fig 1: P1 Error Analysis by MC Range", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_mc_range_error.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig1 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 2: EPO空間のPCA可視化 (train13種 vs test6種)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 2] EPO space PCA: train vs test species...")
# EPO後の全データをPCA
X_all_epo = np.vstack([Xtr_epo, Xte_epo])
pca2 = PCA(n_components=5, random_state=42)
pca2.fit(X_all_epo)
Ztr = pca2.transform(Xtr_epo)   # (1322, 5)
Zte = pca2.transform(Xte_epo)   # (550, 5)

train_sp = sorted(set(sp_train))
cmap_t   = plt.cm.tab20
sp_colors = {s: cmap_t(i/len(train_sp)) for i, s in enumerate(train_sp)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PC1 vs PC2: train種ごとに色分け
for s in train_sp:
    mask = sp_train == s
    axes[0].scatter(Ztr[mask, 0], Ztr[mask, 1], c=[sp_colors[s]], s=15, alpha=0.6,
                    label=f"sp{s}(tr)")
axes[0].scatter(Zte[:, 0], Zte[:, 1], c="black", s=30, alpha=0.7, marker="^",
                label="test (all 6sp)")
axes[0].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("EPO space: PC1 vs PC2  (train colored, test=black^)")
axes[0].legend(fontsize=6, ncol=3, loc="upper right")

# PC1 vs PC3
for s in train_sp:
    mask = sp_train == s
    axes[1].scatter(Ztr[mask, 0], Ztr[mask, 2], c=[sp_colors[s]], s=15, alpha=0.6)
axes[1].scatter(Zte[:, 0], Zte[:, 2], c="black", s=30, alpha=0.7, marker="^")
axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC3 ({pca2.explained_variance_ratio_[2]*100:.1f}%)")
axes[1].set_title("EPO space: PC1 vs PC3")

# test vs train の PC1 分布: どの訓練種に近いか
axes[2].hist(Ztr[:, 0], bins=40, alpha=0.5, color="steelblue", label="train")
axes[2].hist(Zte[:, 0], bins=40, alpha=0.5, color="tomato", label="test")
axes[2].set_xlabel("PC1 score"); axes[2].set_ylabel("Count")
axes[2].set_title("EPO-PC1 distribution: train vs test")
axes[2].legend()

# per-species PC1 range boxplot
all_sp_z = [(sp_train==s, f"sp{s}(tr)") for s in train_sp]
fig2b, ax2b = plt.subplots(figsize=(14, 4))
data_pc1 = [Ztr[sp_train==s, 0] for s in train_sp] + [Zte[:, 0]]
labels_pc1 = [f"sp{s}(tr)" for s in train_sp] + ["test"]
bp = ax2b.boxplot(data_pc1, labels=labels_pc1, patch_artist=True)
for i, (patch, lbl) in enumerate(zip(bp["boxes"], labels_pc1)):
    patch.set_facecolor("tomato" if lbl=="test" else sp_colors.get(int(lbl[2:lbl.index("(")]), "steelblue"))
ax2b.set_ylabel("PC1 score (EPO space)"); ax2b.set_title("PC1 distribution per species (EPO space)")
plt.xticks(rotation=45, ha="right", fontsize=8)
fig2b.tight_layout()
fig2b.savefig(OUT_DIR / "fig2b_epo_pc1_boxplot.png", dpi=120, bbox_inches="tight")
plt.close(fig2b)

# PC1 vs PC2 with MC coloring for train
fig2c, ax2c = plt.subplots(figsize=(9, 6))
sc2 = ax2c.scatter(Ztr[:, 0], Ztr[:, 1], c=y, cmap="plasma", s=15, alpha=0.7, vmin=0, vmax=200)
ax2c.scatter(Zte[:, 0], Zte[:, 1], c="black", s=40, marker="^", alpha=0.6, label="test")
plt.colorbar(sc2, ax=ax2c, label="MC (%)")
ax2c.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax2c.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax2c.set_title("EPO space: train colored by MC, test=black^")
ax2c.legend()
fig2c.tight_layout()
fig2c.savefig(OUT_DIR / "fig2c_epo_mc_color.png", dpi=120, bbox_inches="tight")
plt.close(fig2c)

fig.suptitle("Fig 2: EPO Projected Space Analysis (Train vs Test)", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_epo_pca.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig2 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 3: P1 特徴量重要度スペクトル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 3] Feature importance spectrum (P1)...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 上: 重要度スペクトル
ax = axes[0]
ax.fill_between(wns, imp_norm, alpha=0.7, color="steelblue")
band_labels = [(5187, "OH-comb\n5187", "red"), (6896, "OH-1st\n6896", "red"),
               (8333, "OH-2nd\n8333", "red"), (4760, "Cell-OH\n4760", "gray"),
               (5900, "CH-1st\n5900", "green"), (7082, "H-bond\n7082", "purple")]
ymax = imp_norm.max()
for wn, lbl, col in band_labels:
    ax.axvline(wn, color=col, ls="--", lw=1, alpha=0.7)
    ax.text(wn, ymax*0.8, lbl, fontsize=7, color=col, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))
ax.invert_xaxis()
ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Normalized importance (gain)")
ax.set_title("P1 Feature importance spectrum (LOSO avg)")

# 下: 生スペクトル平均 (比較用)
Xtr_msc = msc(Xtr_raw, ref)
axes[1].plot(wns, Xtr_msc.mean(axis=0), color="steelblue", lw=1, label="Train MSC mean")
for wn, lbl, col in band_labels:
    axes[1].axvline(wn, color=col, ls="--", lw=1, alpha=0.7)
axes[1].invert_xaxis()
axes[1].set_xlabel("Wavenumber (cm⁻¹)"); axes[1].set_ylabel("Absorbance (MSC)")
axes[1].set_title("Reference: MSC mean spectrum"); axes[1].legend()

fig.suptitle("Fig 3: P1 Feature Importance vs Raw Spectrum", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig3 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 4: テスト予測値の分布 vs 訓練樹種
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 4] Test prediction distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y, bins=50, alpha=0.6, color="steelblue", label=f"Train MC (n={len(y)})", density=True)
axes[0].hist(test_preds, bins=50, alpha=0.6, color="tomato",
             label=f"Test pred (n={len(test_preds)})", density=True)
axes[0].axvline(30, color="orange", lw=1, ls="--", label="FSP=30%")
axes[0].set_xlabel("MC (%)"); axes[0].set_ylabel("Density")
axes[0].set_title("Train MC vs Test Predicted MC"); axes[0].legend()

# 樹種別 test 予測統計
print(f"\nTest prediction stats:")
print(f"  min={test_preds.min():.1f}  max={test_preds.max():.1f}  "
      f"mean={test_preds.mean():.1f}  median={np.median(test_preds):.1f}")
print(f"  below FSP (<30%): {(test_preds<30).sum()} ({(test_preds<30).mean()*100:.1f}%)")
print(f"  above 100%:       {(test_preds>100).sum()} ({(test_preds>100).mean()*100:.1f}%)")

# per-species MC boxplot (train)
data_bp  = [y[sp_train==s] for s in train_sp]
lab_bp   = [f"sp{s}" for s in train_sp]
bp2 = axes[1].boxplot(data_bp, labels=lab_bp, patch_artist=True,
                       medianprops=dict(color="red", lw=2))
for patch in bp2["boxes"]:
    patch.set_facecolor("lightblue")
axes[1].hist(test_preds, bins=20, orientation="horizontal", alpha=0.4,
             color="tomato", density=True, label="Test preds dist")
axes[1].set_ylabel("MC (%)"); axes[1].set_xlabel("Species / Density")
axes[1].set_title("Train MC per species + Test pred range (right)")
plt.setp(axes[1].get_xticklabels(), rotation=45, fontsize=7)
axes[1].legend(fontsize=8)

fig.suptitle("Fig 4: Test Prediction Distribution", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_test_pred_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig4 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 5: sp15の特異性分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 5] sp15 anomaly analysis...")
sp15_mask = sp_train == 15
sp15_y    = y[sp15_mask]
sp15_Xsg  = Xtr_sg[sp15_mask]
sp15_Xepo = Xtr_epo[sp15_mask]
sp15_oof  = oof[sp15_mask]
sp15_res  = residuals[sp15_mask]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# sp15 vs 全種平均スペクトル (SG微分後)
ax = axes[0, 0]
ax.plot(wns, Xtr_sg.mean(axis=0), "b-", lw=1, alpha=0.7, label="All train mean")
ax.plot(wns, sp15_Xsg.mean(axis=0), "r-", lw=1.5, label="sp15 mean")
for sp_o in train_sp:
    if sp_o != 15:
        ax.plot(wns, Xtr_sg[sp_train==sp_o].mean(axis=0), "gray", lw=0.5, alpha=0.3)
ax.invert_xaxis()
ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_title("SG derivative: sp15 vs others")
ax.legend()

# sp15 MC分布 vs 他種
ax2 = axes[0, 1]
for s in train_sp:
    if s != 15:
        ax2.hist(y[sp_train==s], bins=20, alpha=0.2, color="steelblue", density=True)
ax2.hist(sp15_y, bins=20, alpha=0.7, color="tomato", density=True, label=f"sp15 (n={sp15_mask.sum()})")
ax2.axvline(30, color="orange", lw=1, ls="--", label="FSP=30%")
ax2.axvline(100, color="purple", lw=1, ls="--", label="MC=100%")
ax2.set_xlabel("MC (%)"); ax2.set_ylabel("Density")
ax2.set_title(f"sp15 MC distribution (max={sp15_y.max():.0f}%, range={sp15_y.min():.0f}-{sp15_y.max():.0f}%)")
ax2.legend()

# sp15 OOF: pred vs actual
ax3 = axes[1, 0]
ax3.scatter(sp15_y, sp15_oof, c=sp15_y, cmap="plasma", s=30, zorder=3)
mx = max(sp15_y.max(), sp15_oof.max())
ax3.plot([0, mx], [0, mx], "r--", lw=1)
ax3.set_xlabel("Actual MC (%)"); ax3.set_ylabel("Predicted MC (%)")
sp15_rmse = np.sqrt(np.mean((sp15_y - sp15_oof)**2))
ax3.set_title(f"sp15 OOF scatter  RMSE={sp15_rmse:.2f}")

# sp15 残差 vs MC
ax4 = axes[1, 1]
ax4.scatter(sp15_y, sp15_res, c=sp15_y, cmap="plasma", s=30)
ax4.axhline(0, color="red", lw=1)
ax4.axvline(30, color="orange", lw=1, ls="--")
ax4.axvline(100, color="purple", lw=1, ls="--")
ax4.set_xlabel("Actual MC (%)"); ax4.set_ylabel("Residual (Pred-Actual)")
ax4.set_title("sp15 residuals vs MC")

fig.suptitle("Fig 5: sp15 Anomaly Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_sp15.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig5 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 6: 残差のスペクトル強度相関 + 高MC域の詳細
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 6] Residual correlation analysis...")
abs_res = np.abs(residuals)
Xtr_msc = msc(Xtr_raw, ref)

# 残差 vs スペクトル平均強度
spec_mean = Xtr_raw.mean(axis=1)
r_spec_res = np.corrcoef(spec_mean, residuals)[0, 1]
r_mc_res   = np.corrcoef(y, residuals)[0, 1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(spec_mean, residuals, c=sp_train, cmap="tab20", alpha=0.5, s=12)
axes[0].axhline(0, color="red", lw=1)
axes[0].set_xlabel("Mean raw absorbance"); axes[0].set_ylabel("Residual")
axes[0].set_title(f"Residual vs Mean absorbance (corr={r_spec_res:.3f})")

axes[1].scatter(y, abs_res, c=sp_train, cmap="tab20", alpha=0.5, s=12)
axes[1].axvline(30, color="orange", lw=1, ls="--")
axes[1].axvline(100, color="purple", lw=1, ls="--")
# 移動平均で傾向線
sort_idx = np.argsort(y)
y_sorted = y[sort_idx]; res_sorted = abs_res[sort_idx]
window_ma = 50
if len(y_sorted) > window_ma:
    ma = np.convolve(res_sorted, np.ones(window_ma)/window_ma, mode="valid")
    axes[1].plot(y_sorted[window_ma//2:-window_ma//2+1], ma, "r-", lw=2, label="Moving avg")
axes[1].set_xlabel("Actual MC (%)"); axes[1].set_ylabel("|Residual|")
axes[1].set_title("Absolute residual vs MC")
axes[1].legend()

# 樹種別の予測バイアス
sp_bias = {s: np.mean(residuals[sp_train==s]) for s in train_sp}
sp_rmse_d = {s: np.sqrt(np.mean((y[sp_train==s]-oof[sp_train==s])**2)) for s in train_sp}
colors_b = ["crimson" if v > 0 else "steelblue" for v in sp_bias.values()]
axes[2].bar([f"sp{s}" for s in train_sp], list(sp_bias.values()), color=colors_b)
axes[2].axhline(0, color="black", lw=1)
axes[2].set_xlabel("Species"); axes[2].set_ylabel("Bias (Pred-Actual)")
axes[2].set_title("P1: Per-species bias")
plt.setp(axes[2].get_xticklabels(), rotation=45, fontsize=8)

fig.suptitle("Fig 6: Residual Structure Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_residual_structure.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  -> fig6 saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テキストサマリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("EDA SUMMARY: P1 Deep Analysis")
print("=" * 65)

print(f"\n[P1 Model]  LOSO-RMSE = {rmse:.4f}  (target: {P1_LOSO:.4f})")
print(f"  avg_iter = {avg_iter}")
print(f"  Overall Bias = {np.mean(residuals):+.4f}")

print(f"\n[Per-species RMSE and Bias]")
header = f"  {'sp':>4}  {'n':>4}  {'MC_range':>12}  {'RMSE':>7}  {'Bias':>8}  {'note':>6}"
print(header)
for s in sorted(train_sp):
    mask = sp_train == s
    n    = mask.sum()
    rmse_s = np.sqrt(np.mean((y[mask]-oof[mask])**2))
    bias_s = np.mean(residuals[mask])
    mc_min, mc_max = y[mask].min(), y[mask].max()
    flag = " (*)" if rmse_s > rmse * 1.5 else ""
    print(f"  sp{s:02d}  {n:>4}  [{mc_min:5.0f}-{mc_max:5.0f}]%  {rmse_s:7.2f}  {bias_s:+8.2f}{flag}")

print(f"\n[MC range RMSE]")
for b, lbl in enumerate(labels_mc):
    mask = bin_idx == b
    if mask.sum() > 0:
        r = np.sqrt(np.mean((y[mask]-oof[mask])**2))
        print(f"  MC {lbl:>7}%: n={mask.sum():>4}  RMSE={r:.2f}")

print(f"\n[Feature importance] Top-15 wavenumbers:")
top15 = np.argsort(imp_norm)[-15:][::-1]
for rank, idx in enumerate(top15):
    print(f"  {rank+1:2d}. wn={wns[idx]:.1f} cm⁻¹  imp={imp_norm[idx]:.4f}")

print(f"\n[EPO space PCA] explained variance: {pca2.explained_variance_ratio_[:5]*100}")

print(f"\n[Test predictions]")
print(f"  n={len(test_preds)}  min={test_preds.min():.1f}  max={test_preds.max():.1f}  "
      f"mean={test_preds.mean():.1f}  median={np.median(test_preds):.1f}")
print(f"  below FSP (<30%): {(test_preds<30).sum()} ({(test_preds<30).mean()*100:.1f}%)")
print(f"  30-100%: {((test_preds>=30)&(test_preds<100)).sum()}")
print(f"  above 100%: {(test_preds>=100).sum()} ({(test_preds>=100).mean()*100:.1f}%)")

print(f"\n[sp15 analysis]")
print(f"  MC range: {sp15_y.min():.0f}–{sp15_y.max():.0f}%  n={sp15_mask.sum()}")
sp15_rmse_low  = np.sqrt(np.mean((sp15_y[sp15_y<30]-sp15_oof[sp15_y<30])**2)) if (sp15_y<30).any() else np.nan
sp15_rmse_mid  = np.sqrt(np.mean((sp15_y[(sp15_y>=30)&(sp15_y<100)]-sp15_oof[(sp15_y>=30)&(sp15_y<100)])**2)) if ((sp15_y>=30)&(sp15_y<100)).any() else np.nan
sp15_rmse_high = np.sqrt(np.mean((sp15_y[sp15_y>=100]-sp15_oof[sp15_y>=100])**2)) if (sp15_y>=100).any() else np.nan
print(f"  RMSE breakdown:")
print(f"    MC<30:    {sp15_rmse_low:.2f}  (n={( sp15_y<30).sum()})")
print(f"    MC 30-100: {sp15_rmse_mid:.2f}  (n={(( sp15_y>=30)&(sp15_y<100)).sum()})")
print(f"    MC>=100:  {sp15_rmse_high:.2f}  (n={(sp15_y>=100).sum()})")

print(f"\n[Residual correlations]")
print(f"  corr(MC, residual)         = {r_mc_res:.4f}")
print(f"  corr(mean_absorbance, res) = {r_spec_res:.4f}")

print(f"\n[Output] Saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
