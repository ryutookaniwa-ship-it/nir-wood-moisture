"""
EDA: Exp-T ベストモデル詳細分析
===================================
分析観点:
  1. ターゲット分布 (全体 / 樹種別 / FSP境界)
  2. 生スペクトル可視化 (樹種別平均 ± std)
  3. 前処理パイプライン比較 (Raw → MSC → MSC+SG)
  4. OOF散布図 + 残差分布
  5. 樹種別 RMSE / Bias / サンプル数
  6. 特徴量重要度スペクトル (全fold平均)
  7. MCレンジ別誤差 (0-30 / 30-100 / 100+)
  8. 難しいサンプルの特定 (|残差| Top-20)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import lightgbm as lgb

sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse, LGBM_BASE_PARAMS
)

OUT_DIR = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\eda_T")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. データ読み込み・前処理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
data        = load_data()
y           = data["y_train"]
X_raw       = data["X_train_raw"]
sp          = data["sp_train"]
wns         = data["wns"]

train_df = pd.read_csv(
    r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\train (1).csv",
    encoding="shift-jis"
)
sp_name_map = dict(zip(train_df["species number"].values, train_df.iloc[:, 2].values))

ref    = X_raw.mean(axis=0)
X_msc  = msc(X_raw, ref)
X_pp   = sg_deriv(X_msc, window=5, polyorder=3)
y_sqrt = np.sqrt(y)

params = {
    **LGBM_BASE_PARAMS,
    "learning_rate":    0.02,
    "num_leaves":       63,
    "feature_fraction": 0.07,
    "min_child_samples": 10,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LOSO-CV: OOF + 特徴量重要度収集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Running LOSO-CV (T params)...")
oof             = np.zeros(len(y))
imp_gain_accum  = np.zeros(len(wns))   # 全fold平均特徴量重要度
fold_results    = []   # (sp_id, rmse, bias, n_samples)

for tr_idx, va_idx, sp_id in loso_folds(sp):
    dtrain = lgb.Dataset(X_pp[tr_idx], label=y_sqrt[tr_idx])
    dval   = lgb.Dataset(X_pp[va_idx], label=y_sqrt[va_idx], reference=dtrain)
    model  = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    preds_sqrt = np.clip(model.predict(X_pp[va_idx]), 0, None)
    oof[va_idx] = preds_sqrt ** 2

    # 特徴量重要度を積算
    imp = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    for fname, iv in zip(feat_names, imp):
        idx = int(fname.split("_")[1])
        if idx < len(wns):
            imp_gain_accum[idx] += iv

    # fold統計
    y_va   = y[va_idx]
    y_pred = oof[va_idx]
    rmse   = float(np.sqrt(np.mean((y_va - y_pred) ** 2)))
    bias   = float(np.mean(y_pred - y_va))
    fold_results.append((sp_id, rmse, bias, len(va_idx)))
    print(f"  Species {sp_id:2d}: n={len(va_idx):3d}  RMSE={rmse:6.2f}  Bias={bias:+7.2f}")

overall_rmse = loso_rmse(oof, y)
print(f"\nOverall LOSO-RMSE = {overall_rmse:.4f}")

imp_norm = imp_gain_accum / (imp_gain_accum.sum() + 1e-9)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 1: ターゲット分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Fig 1] Target distribution...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 全体ヒストグラム
axes[0].hist(y, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
axes[0].axvline(30, color="red", ls="--", lw=1.5, label="FSP ≈ 30%")
axes[0].set_xlabel("Moisture Content (%)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Overall MC distribution (n={len(y)})")
axes[0].legend()

# sqrt(y)変換後
axes[1].hist(y_sqrt, bins=50, color="salmon", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("√MC")
axes[1].set_title("After √ transform (target for LGBM)")

# 樹種別 box plot
species_ids = sorted(set(sp))
data_by_sp  = [y[sp == s] for s in species_ids]
labels      = [f"{s}\n({sp_name_map.get(s,'?')[:4]})" for s in species_ids]
bp = axes[2].boxplot(data_by_sp, labels=labels, patch_artist=True,
                     medianprops=dict(color="red", lw=2))
for patch in bp["boxes"]:
    patch.set_facecolor("lightblue")
axes[2].axhline(30, color="red", ls="--", lw=1, label="FSP")
axes[2].set_xlabel("Species")
axes[2].set_ylabel("MC (%)")
axes[2].set_title("MC distribution per species")
axes[2].legend(fontsize=8)
plt.xticks(rotation=45)

fig.suptitle("Fig 1: Target Variable Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_target_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 2: 生スペクトル (樹種別平均)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 2] Raw spectra by species...")
fig, ax = plt.subplots(figsize=(14, 5))
cmap = plt.cm.tab20
for i, s in enumerate(species_ids):
    mask = sp == s
    mu   = X_raw[mask].mean(axis=0)
    ax.plot(wns, mu, color=cmap(i / len(species_ids)), lw=1,
            label=f"{s}:{sp_name_map.get(s,'?')[:6]}")
# 主要吸収帯マーカー
for wn, label in [(5187, "OH comb.\n5187"), (6896, "OH 1st\n6896"), (8333, "OH 2nd\n8333")]:
    ax.axvline(wn, color="red", ls=":", lw=1)
    ax.text(wn, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.95,
            label, fontsize=7, color="red", ha="center")
ax.invert_xaxis()
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Absorbance")
ax.set_title("Fig 2: Mean raw spectra per species")
ax.legend(fontsize=7, ncol=3, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_raw_spectra.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 3: 前処理パイプライン比較 (1樹種の代表サンプル)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 3] Preprocessing pipeline comparison...")
s_rep  = species_ids[0]
idx_s  = np.where(sp == s_rep)[0][:5]   # 最初の5サンプル

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
titles = ["Raw", "After MSC", "After MSC + SG(w=5,p=3)"]
Xs     = [X_raw, X_msc, X_pp]
for ax, Xi, title in zip(axes, Xs, titles):
    for i in idx_s:
        ax.plot(wns, Xi[i], alpha=0.7, lw=0.9)
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_title(title)
    for wn in [5187, 6896, 8333]:
        ax.axvline(wn, color="red", ls=":", lw=0.8)

fig.suptitle(f"Fig 3: Preprocessing pipeline  (Species={s_rep}, 5 samples)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_preprocessing.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 4: OOF散布図 + 残差分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 4] OOF scatter + residuals...")
residuals = oof - y
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Predicted vs Actual (色: 樹種)
sc = axes[0].scatter(y, oof, c=sp, cmap="tab20", alpha=0.5, s=12)
mx = max(y.max(), oof.max())
axes[0].plot([0, mx], [0, mx], "r--", lw=1)
axes[0].set_xlabel("Actual MC (%)")
axes[0].set_ylabel("Predicted MC (%)")
axes[0].set_title("OOF: Predicted vs Actual")
fig.colorbar(sc, ax=axes[0], label="Species")

# Residuals vs Actual
axes[1].scatter(y, residuals, c=sp, cmap="tab20", alpha=0.5, s=12)
axes[1].axhline(0, color="red", lw=1)
axes[1].axvline(30, color="orange", lw=1, ls="--", label="FSP=30%")
axes[1].set_xlabel("Actual MC (%)")
axes[1].set_ylabel("Predicted - Actual")
axes[1].set_title("Residuals vs Actual")
axes[1].legend(fontsize=8)

# 残差ヒストグラム
axes[2].hist(residuals, bins=50, color="steelblue", edgecolor="white", lw=0.4)
axes[2].axvline(0, color="red", lw=1)
axes[2].axvline(np.mean(residuals), color="orange", lw=1.5,
                label=f"Mean={np.mean(residuals):.2f}")
axes[2].set_xlabel("Residual (Pred - Actual)")
axes[2].set_title(f"Residual distribution  RMSE={overall_rmse:.2f}")
axes[2].legend()

fig.suptitle("Fig 4: OOF Prediction Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_oof_scatter.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 5: 樹種別 RMSE / Bias / サンプル数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 5] Per-species metrics...")
fr_df = pd.DataFrame(fold_results, columns=["species", "rmse", "bias", "n"])
fr_df["sp_label"] = fr_df["species"].map(
    lambda s: f"{s}\n{sp_name_map.get(s,'?')[:6]}"
)
fr_df = fr_df.sort_values("rmse", ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors_rmse = ["crimson" if r > overall_rmse * 1.3 else "steelblue" for r in fr_df["rmse"]]
axes[0].bar(fr_df["sp_label"], fr_df["rmse"], color=colors_rmse)
axes[0].axhline(overall_rmse, color="red", ls="--", lw=1.5, label=f"Overall={overall_rmse:.2f}")
axes[0].set_title("Per-species RMSE (sorted desc)")
axes[0].set_ylabel("RMSE")
axes[0].legend(fontsize=8)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)

colors_bias = ["crimson" if b > 0 else "steelblue" for b in fr_df["bias"]]
axes[1].bar(fr_df["sp_label"], fr_df["bias"], color=colors_bias)
axes[1].axhline(0, color="black", lw=1)
axes[1].set_title("Per-species Bias (Pred - Actual)")
axes[1].set_ylabel("Mean Bias (%)")
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)

axes[2].scatter(fr_df["n"], fr_df["rmse"], s=80, c=fr_df["species"], cmap="tab20", zorder=3)
for _, row in fr_df.iterrows():
    axes[2].annotate(str(int(row["species"])), (row["n"], row["rmse"]),
                     fontsize=7, ha="left", va="bottom")
axes[2].axhline(overall_rmse, color="red", ls="--", lw=1)
axes[2].set_xlabel("N samples in species")
axes[2].set_ylabel("RMSE")
axes[2].set_title("RMSE vs sample count per species")

fig.suptitle("Fig 5: Per-species Error Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_species_error.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 6: 特徴量重要度スペクトル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 6] Feature importance spectrum...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(wns, imp_norm, alpha=0.7, color="steelblue", label="Importance (gain, norm.)")

# 既知吸収帯のアノテーション
band_labels = [
    (5187, "OH comb.\n5187", "red"),
    (6896, "OH 1st\n6896",   "red"),
    (8333, "OH 2nd\n8333",   "red"),
    (4760, "Cell. OH+CH\n4760", "gray"),
    (5900, "CH 1st\n5900",   "green"),
]
ymax = imp_norm.max()
for wn, lbl, col in band_labels:
    ax.axvline(wn, color=col, ls="--", lw=1, alpha=0.7)
    ax.text(wn, ymax * 0.85, lbl, fontsize=7, color=col, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5))

ax.invert_xaxis()
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Normalized importance")
ax.set_title(f"Fig 6: Feature importance spectrum (LOSO avg, top signal peaks)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig6_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 7: MCレンジ別誤差
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 7] Error by MC range...")
bins   = [0, 30, 60, 100, 150, 200, np.inf]
labels_r = ["0-30%\n(below FSP)", "30-60%", "60-100%", "100-150%", "150-200%", "200%+"]
bin_idx  = np.digitize(y, bins) - 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rmse_by_bin = []
n_by_bin    = []
for b in range(len(labels_r)):
    mask = bin_idx == b
    if mask.sum() > 0:
        rmse_by_bin.append(np.sqrt(np.mean((y[mask] - oof[mask]) ** 2)))
        n_by_bin.append(mask.sum())
    else:
        rmse_by_bin.append(0)
        n_by_bin.append(0)

x = np.arange(len(labels_r))
colors_r = ["salmon" if r > overall_rmse else "steelblue" for r in rmse_by_bin]
bars = axes[0].bar(x, rmse_by_bin, color=colors_r)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_r, fontsize=8)
axes[0].axhline(overall_rmse, color="red", ls="--", lw=1.5, label=f"Overall={overall_rmse:.2f}")
axes[0].set_ylabel("RMSE")
axes[0].set_title("RMSE by MC range")
axes[0].legend()
for bar, n in zip(bars, n_by_bin):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"n={n}", ha="center", fontsize=7)

# Bias by range
bias_by_bin = []
for b in range(len(labels_r)):
    mask = bin_idx == b
    if mask.sum() > 0:
        bias_by_bin.append(np.mean(oof[mask] - y[mask]))
    else:
        bias_by_bin.append(0)

colors_b = ["crimson" if b > 0 else "steelblue" for b in bias_by_bin]
axes[1].bar(x, bias_by_bin, color=colors_b)
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_r, fontsize=8)
axes[1].axhline(0, color="black", lw=1)
axes[1].set_ylabel("Mean Bias (Pred - Actual)")
axes[1].set_title("Bias by MC range")

fig.suptitle("Fig 7: Error Analysis by Moisture Content Range", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_error_by_range.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 8: 難しいサンプル Top-20
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[Fig 8] Hard samples analysis...")
abs_res = np.abs(residuals)
top20   = np.argsort(abs_res)[-20:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(
    [f"sp{sp[i]} MC={y[i]:.0f}" for i in top20],
    abs_res[top20], color="salmon"
)
axes[0].set_xlabel("|Residual| (%)")
axes[0].set_title("Top-20 hardest samples (|Residual|)")
axes[0].invert_yaxis()

# 難しいサンプルのスペクトル vs 同樹種の平均スペクトル
ax2 = axes[1]
for rank, i in enumerate(top20[:5]):
    s_id   = sp[i]
    sp_mean = X_pp[sp == s_id].mean(axis=0)
    delta  = X_pp[i] - sp_mean
    ax2.plot(wns, delta, lw=0.8, alpha=0.7,
             label=f"sp{s_id} MC={y[i]:.0f} err={residuals[i]:+.0f}")
ax2.axhline(0, color="black", lw=0.5)
for wn in [5187, 6896, 8333]:
    ax2.axvline(wn, color="red", ls=":", lw=0.8)
ax2.invert_xaxis()
ax2.set_xlabel("Wavenumber (cm⁻¹)")
ax2.set_ylabel("Δ (sample - species mean)")
ax2.set_title("Hard samples: spectrum deviation from species mean (top 5)")
ax2.legend(fontsize=7)

fig.suptitle("Fig 8: Hard Sample Analysis", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig8_hard_samples.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テキストサマリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("EDA SUMMARY - Exp T (MSC+SG w=5,p=3 + sqrt + LGBM)")
print("=" * 60)

print(f"\n[Target]")
print(f"  N = {len(y)}, range = {y.min():.1f} – {y.max():.1f} %")
print(f"  Below FSP (<30%): {(y < 30).sum()} ({(y < 30).mean()*100:.1f}%)")
print(f"  Above FSP (≥30%): {(y >= 30).sum()} ({(y >= 30).mean()*100:.1f}%)")

print(f"\n[Model]  LOSO-RMSE = {overall_rmse:.4f}")
print(f"  Overall Bias = {np.mean(residuals):+.4f}")
print(f"  |Residual| 90th pctile = {np.percentile(abs_res, 90):.2f}")

print(f"\n[Per-species RMSE]")
for _, row in fr_df.iterrows():
    flag = "⚠" if row["rmse"] > overall_rmse * 1.5 else " "
    print(f"  {flag} sp{int(row['species']):2d} ({sp_name_map.get(int(row['species']),'?')[:8]:<8}): "
          f"RMSE={row['rmse']:6.2f}  Bias={row['bias']:+7.2f}  n={int(row['n'])}")

print(f"\n[Feature importance] Top-10 wavelengths:")
top10_idx = np.argsort(imp_norm)[-10:][::-1]
for rank, idx in enumerate(top10_idx):
    print(f"  {rank+1:2d}. {wns[idx]:.1f} cm⁻¹  imp={imp_norm[idx]:.4f}")

print(f"\n[Output]  Figures saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
