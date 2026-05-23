"""
EDA: 4つの多面的分析
  1. 樹種別RMSE (LOSO-CV)
  2. 残差プロット（含水率レンジ別）
  3. 特徴量重要度のスペクトルマッピング
  4. 前処理別LOSO-RMSE比較 (Raw / SNV / SNV+SG1)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/train (1).csv"
OUT_DIR    = "C:/Users/ryuch/OneDrive/\u30c7\u30b9\u30af\u30c8\u30c3\u30d7/my_kaggle_project/output/eda"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, encoding='shift-jis')
target_col = train.columns[3]
spec_cols  = train.columns[4:].tolist()
wns        = np.array([float(c) for c in spec_cols])

y          = train[target_col].values
X_raw      = train[spec_cols].values.astype(np.float64)
sp         = train['species number'].values
species_list = sorted(set(sp))

# 樹種番号→名前マップ（コンペ記載の19種、numbering推定）
# 記載順: イチョウ,クスノキ,ウエンジ,ウォールナット,クリ,ケヤキ,スギ,スプルース,
#         タモ,チーク,チェリー,トチ,ナラ,ヒノキ,ベイスギ,米ヒバ,ベイマツ,ヤマザクラ,ホワイトオーク
sp_names = {
    1:'イチョウ',2:'クスノキ',3:'ウエンジ',4:'ウォールナット',5:'クリ',
    6:'ケヤキ',7:'スギ',8:'スプルース',9:'タモ',10:'チーク',
    11:'チェリー',12:'トチ',13:'ナラ',14:'ヒノキ',15:'ベイスギ',
    16:'米ヒバ',17:'ベイマツ',18:'ヤマザクラ',19:'ホワイトオーク'
}

# ── Preprocessing ──────────────────────────────────────────────────────────────
def snv(X):
    m = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True)
    return (X - m) / np.where(s == 0, 1, s)

def sg1(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)

X_snv_sg1 = sg1(snv(X_raw))
X_snv     = snv(X_raw)

# ── LGBM共通パラメータ ─────────────────────────────────────────────────────────
lgbm_params = dict(
    objective='regression', metric='rmse', verbosity=-1,
    n_jobs=-1, random_state=42,
    learning_rate=0.05, num_leaves=31,
    feature_fraction=0.1, min_child_samples=10,
)

def run_loso(X, y, sp, params, n_rounds=500, return_oof=True):
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    for s in sorted(set(sp)):
        tr_idx = np.where(sp != s)[0]
        va_idx = np.where(sp == s)[0]
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        dval   = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=n_rounds,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
        oof[va_idx] = model.predict(X[va_idx])
        importances += model.feature_importance(importance_type='gain')
    rmse = np.sqrt(np.mean((y - oof) ** 2))
    return oof, importances / len(set(sp)), rmse

# ── 1. 樹種別RMSE ──────────────────────────────────────────────────────────────
print("Running LOSO-CV for EDA (SNV+SG1)...")
oof, importances, loso_rmse = run_loso(X_snv_sg1, y, sp, lgbm_params)
print(f"  Overall LOSO-RMSE: {loso_rmse:.4f}")

sp_rmse = {}
sp_bias = {}
sp_n    = {}
for s in species_list:
    mask = sp == s
    err  = oof[mask] - y[mask]
    sp_rmse[s] = np.sqrt(np.mean(err**2))
    sp_bias[s] = err.mean()
    sp_n[s]    = mask.sum()
    print(f"  species={s:2d} ({sp_names.get(s,'?'):8s}): n={sp_n[s]:3d}  "
          f"RMSE={sp_rmse[s]:.2f}  bias={sp_bias[s]:+.2f}")

# Plot 1: 樹種別RMSE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sp_labels = [f"{sp_names.get(s,'?')}\n(sp{s})" for s in species_list]
rmse_vals = [sp_rmse[s] for s in species_list]
bias_vals = [sp_bias[s] for s in species_list]
colors    = ['#d62728' if r > loso_rmse*1.3 else '#1f77b4' for r in rmse_vals]

ax = axes[0]
bars = ax.bar(sp_labels, rmse_vals, color=colors)
ax.axhline(loso_rmse, color='k', linestyle='--', label=f'Overall RMSE={loso_rmse:.2f}')
ax.set_title('1. Species-level RMSE (LOSO-CV, SNV+SG1)', fontsize=12)
ax.set_ylabel('RMSE'); ax.set_xticklabels(sp_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}',
            ha='center', va='bottom', fontsize=7)

ax = axes[1]
colors2 = ['#d62728' if b > 0 else '#2ca02c' for b in bias_vals]
ax.bar(sp_labels, bias_vals, color=colors2)
ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
ax.set_title('1b. Species-level Bias (pred - true)', fontsize=12)
ax.set_ylabel('Bias (%)'); ax.set_xticklabels(sp_labels, rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/1_species_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_species_rmse.png")

# ── 2. 残差プロット ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 2a: 予測 vs 真値
ax = axes[0]
lims = [0, max(y.max(), oof.max()) * 1.05]
ax.scatter(y, oof, alpha=0.3, s=15, c=sp, cmap='tab20')
ax.plot(lims, lims, 'r--', linewidth=1, label='perfect')
ax.set_xlabel('True MC (%)'); ax.set_ylabel('Predicted MC (%)')
ax.set_title('2a. Predicted vs True', fontsize=12)
ax.set_xlim(lims); ax.set_ylim(lims); ax.legend()

# 2b: 残差 vs 真値
residuals = oof - y
ax = axes[1]
ax.scatter(y, residuals, alpha=0.3, s=15, c=sp, cmap='tab20')
ax.axhline(0, color='r', linestyle='--', linewidth=1)
# FSP線
ax.axvline(30, color='orange', linestyle=':', linewidth=1.5, label='FSP 30%')
ax.set_xlabel('True MC (%)'); ax.set_ylabel('Residual (pred - true)')
ax.set_title('2b. Residuals vs True MC', fontsize=12); ax.legend()

# 2c: MC帯別RMSE
bins  = [0, 10, 20, 30, 50, 80, 120, 200, 350]
bin_rmse, bin_n, bin_mid = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (y >= lo) & (y < hi)
    if mask.sum() > 0:
        bin_rmse.append(np.sqrt(np.mean(residuals[mask]**2)))
        bin_n.append(mask.sum())
        bin_mid.append((lo+hi)/2)

ax = axes[2]
ax2 = ax.twinx()
ax.bar([f'{bins[i]}-{bins[i+1]}' for i in range(len(bin_mid))],
       bin_rmse, alpha=0.7, color='steelblue', label='RMSE')
ax2.plot([f'{bins[i]}-{bins[i+1]}' for i in range(len(bin_mid))],
         bin_n, 'o-', color='orange', label='n samples')
ax.set_xlabel('MC range (%)'); ax.set_ylabel('RMSE', color='steelblue')
ax2.set_ylabel('n samples', color='orange')
ax.set_title('2c. RMSE by MC range', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/2_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 2_residuals.png")

# ── 3. 特徴量重要度のスペクトルマッピング ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 参考: 代表スペクトル（全サンプル平均）
mean_spec     = X_snv_sg1.mean(axis=0)
mean_spec_snv = X_snv.mean(axis=0)

# 重要度（gain）を正規化
imp_norm = importances / importances.max()

# 水の主要吸収帯
water_bands = [(5087,5287,'#ff7f0e','5187 O-H comb.'),
               (6746,7046,'#2ca02c','6896 O-H 1st ov.'),
               (8133,8533,'#9467bd','8333 O-H 2nd ov.')]

ax = axes[0]
ax.fill_between(wns, imp_norm, alpha=0.6, color='steelblue', label='Feature importance (gain, norm)')
for lo, hi, c, lbl in water_bands:
    ax.axvspan(lo, hi, alpha=0.2, color=c, label=lbl)
ax.set_xlabel('Wavenumber (cm-1)'); ax.set_ylabel('Normalized Importance')
ax.set_title('3. Feature Importance mapped to Spectrum (LGBM, SNV+SG1, LOSO)', fontsize=12)
ax.invert_xaxis(); ax.legend(fontsize=8)

ax = axes[1]
# 累積重要度 top-k% の波数がどの帯域に集中するか
sorted_idx = np.argsort(importances)[::-1]
cum_imp    = np.cumsum(importances[sorted_idx]) / importances.sum()
top50_mask = np.zeros(len(wns), dtype=bool)
top50_mask[sorted_idx[:np.searchsorted(cum_imp, 0.5)]] = True
top80_mask = np.zeros(len(wns), dtype=bool)
top80_mask[sorted_idx[:np.searchsorted(cum_imp, 0.8)]] = True

ax2 = ax.twinx()
ax.plot(wns, mean_spec_snv, color='gray', alpha=0.5, linewidth=0.8, label='Mean spectrum (SNV)')
ax2.fill_between(wns, imp_norm, alpha=0.4, color='steelblue')
ax2.fill_between(wns, imp_norm * top50_mask, alpha=0.6, color='red', label='Top 50% importance')
for lo, hi, c, lbl in water_bands:
    ax.axvspan(lo, hi, alpha=0.15, color=c)
ax.set_xlabel('Wavenumber (cm-1)'); ax.set_ylabel('Mean SNV spectrum', color='gray')
ax2.set_ylabel('Normalized Importance', color='steelblue')
ax.set_title('3b. Top-50% importance wavenumbers vs Mean spectrum', fontsize=12)
ax.invert_xaxis(); ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)

# 帯域別の重要度集計
print("\n  Feature importance by spectral region:")
regions = [
    ('9000-8000', 9000, 8000),
    ('8000-7000', 8000, 7000),
    ('7000-6000', 7000, 6000),
    ('6000-5000', 6000, 5000),
    ('5000-4000', 5000, 4000),
]
total_imp = importances.sum()
for name, hi, lo in regions:
    mask = (wns >= lo) & (wns <= hi)
    frac = importances[mask].sum() / total_imp
    print(f"    {name} cm-1: {frac*100:.1f}%")

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 3_feature_importance.png")

# ── 4. 前処理別LOSO-RMSE比較 ──────────────────────────────────────────────────
print("\nRunning LOSO-CV for preprocessing comparison...")

preproc_configs = {
    'Raw':         X_raw,
    'SNV':         X_snv,
    'SNV+SG1':     X_snv_sg1,
    'SNV+SG1\n(log y)': X_snv_sg1,   # 対数変換あり
}

results = {}
oofs    = {}
for name, X in preproc_configs.items():
    if 'log' in name:
        y_log = np.log1p(y)
        oof_log, _, rmse_log = run_loso(X, y_log, sp, lgbm_params)
        oof_orig = np.expm1(oof_log)
        rmse = np.sqrt(np.mean((y - oof_orig)**2))
        oof_use = oof_orig
    else:
        oof_use, _, rmse = run_loso(X, y, sp, lgbm_params)
    results[name] = rmse
    oofs[name]    = oof_use
    print(f"  [{name:15s}]  LOSO-RMSE={rmse:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
names = list(results.keys())
rmses = list(results.values())
colors4 = ['#d62728' if r == min(rmses) else '#1f77b4' for r in rmses]
bars = ax.bar(names, rmses, color=colors4)
ax.axhline(21.48, color='gray', linestyle='--', label='Baseline G (21.48)')
ax.set_title('4. Preprocessing Comparison (LOSO-CV)', fontsize=12)
ax.set_ylabel('LOSO-RMSE'); ax.legend()
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# 4b: 前処理別の予測 vs 真値（散布図比較）
ax = axes[1]
for name, oof_use in oofs.items():
    ax.scatter(y, oof_use, alpha=0.15, s=10, label=f'{name} ({results[name]:.1f})')
lims = [0, max(y.max(), max(o.max() for o in oofs.values()))*1.05]
ax.plot(lims, lims, 'k--', linewidth=1)
ax.axvline(30, color='orange', linestyle=':', linewidth=1, label='FSP')
ax.set_xlabel('True MC (%)'); ax.set_ylabel('Predicted MC (%)')
ax.set_title('4b. Predicted vs True by preprocessing', fontsize=12)
ax.set_xlim(lims); ax.set_ylim(lims); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/4_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 4_preprocessing_comparison.png")

# ── 5. 含水率範囲別残差の前処理比較 ──────────────────────────────────────────
fig, axes = plt.subplots(1, len(oofs), figsize=(18, 4))
for ax, (name, oof_use) in zip(axes, oofs.items()):
    res = oof_use - y
    ax.scatter(y, res, alpha=0.25, s=10, color='steelblue')
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.axvline(30, color='orange', linestyle=':', linewidth=1.5, label='FSP')
    # 移動平均残差
    sort_idx = np.argsort(y)
    y_s, r_s = y[sort_idx], res[sort_idx]
    window = 50
    if len(y_s) > window:
        moving_avg = np.convolve(r_s, np.ones(window)/window, mode='valid')
        ax.plot(y_s[window//2:-window//2+1], moving_avg, 'r-', linewidth=2, label='Moving avg')
    ax.set_xlabel('True MC (%)'); ax.set_ylabel('Residual')
    ax.set_title(f'{name}\nRMSE={results[name]:.2f}', fontsize=10)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/5_residuals_by_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 5_residuals_by_preprocessing.png")

# ── 最終サマリー ───────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"\n[1] 樹種別RMSE (worst -> best):")
for s, r in sorted(sp_rmse.items(), key=lambda x: -x[1]):
    print(f"    {sp_names.get(s,'?'):10s}: RMSE={r:.2f}  bias={sp_bias[s]:+.2f}  n={sp_n[s]}")

print(f"\n[2] 含水率帯別RMSE:")
for i in range(len(bin_mid)):
    lo, hi = bins[i], bins[i+1]
    print(f"    {lo:3d}-{hi:3d}%: RMSE={bin_rmse[i]:.2f}  n={bin_n[i]}")

print(f"\n[3] 特徴量重要度 帯域別集計:")
for name, hi, lo in regions:
    mask = (wns >= lo) & (wns <= hi)
    frac = importances[mask].sum() / total_imp
    print(f"    {name}: {frac*100:.1f}%")

print(f"\n[4] 前処理別LOSO-RMSE:")
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    delta = rmse - 21.48
    print(f"    {name:20s}: {rmse:.4f}  (vs baseline: {delta:+.4f})")

print(f"\n全グラフを {OUT_DIR} に保存済み")
