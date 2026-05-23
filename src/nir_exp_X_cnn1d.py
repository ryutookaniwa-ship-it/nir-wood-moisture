"""
Experiment X: 1D-CNN (浅いアーキテクチャ)
==============================================
仮説:
  LightGBMは各波長を独立に評価するため、スペクトルの「局所的な波形形状」
  （ピークの幅・傾き・周辺との関係）を捉えられない。
  Conv1Dは隣接波長の局所相関を直接学習できる。

アーキテクチャ (意図的に浅く設計: N=1322 の小データ対策):
  Input (1555,) → Reshape (1555,1)
  → Conv1D(32, k=15) → BN → ReLU → MaxPool1D(4)  → (388, 32)
  → Conv1D(32, k=9)  → BN → ReLU → GlobalAvgPool  → (32,)
  → Linear(64) → Dropout(0.4) → ReLU → Linear(1)

ターゲット: sqrt(y) 変換 (T-best と統一)
前処理: MSC + SG(w=5, poly=3) (T-best と統一)
評価: LOSO-CV (Leave-One-Species-Out)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, plot_residuals,
)
import warnings; warnings.filterwarnings("ignore")

EXP_LETTER = "X"
OUT_PATH   = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\submission_X_cnn1d.csv"
DEVICE     = "cpu"
SEED       = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── アーキテクチャ ────────────────────────────────────────────────────────────
class SpectralCNN(nn.Module):
    def __init__(self, n_features: int = 1555):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),                          # → (32, 388)
            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # Global Average Pooling → (32,)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1555)
        x = x.unsqueeze(1)                 # (B, 1, 1555)
        x = self.conv(x)                   # (B, 32, L)
        x = x.mean(dim=2)                  # Global Average Pool → (B, 32)
        return self.head(x).squeeze(1)     # (B,)


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 200, patience: int = 20,
    batch_size: int = 64, lr: float = 1e-3,
) -> tuple:
    """1 fold を訓練して val RMSE と予測を返す。"""
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    model = SpectralCNN(n_features=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_preds    = None
    wait          = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t).numpy()
            val_loss  = float(np.mean((val_preds - y_va_t.numpy()) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds    = val_preds.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_preds


def train_final(
    X_tr: np.ndarray, y_tr: np.ndarray,
    n_rounds: int = 150,
    batch_size: int = 64, lr: float = 1e-3,
) -> SpectralCNN:
    """全訓練データで最終モデルを訓練。"""
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)
    model = SpectralCNN(n_features=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_rounds)
    loss_fn = nn.MSELoss()
    for _ in range(n_rounds):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
    return model


# ── データ・前処理 ────────────────────────────────────────────────────────────
data = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
X_tr   = sg_deriv(msc(X_train_raw, ref), window=5, polyorder=3).astype(np.float32)
X_te   = sg_deriv(msc(X_test_raw,  ref), window=5, polyorder=3).astype(np.float32)
y_sqrt = np.sqrt(y_train).astype(np.float32)

# ── 特徴量スケーリング (CNN は勾配ベースなので標準化必須) ─────────────────────
mu  = X_tr.mean(axis=0)
sig = X_tr.std(axis=0) + 1e-8
X_tr_sc = (X_tr - mu) / sig
X_te_sc = (X_te - mu) / sig

print("=== Experiment X: 1D-CNN (LOSO-CV) ===")
print(f"Input: {X_tr_sc.shape[1]} features | Device: {DEVICE}")
print()

# ── LOSO-CV ──────────────────────────────────────────────────────────────────
oof = np.zeros(len(y_train))

for fold_i, (tr_idx, va_idx, sp) in enumerate(loso_folds(sp_train)):
    preds_sqrt = train_one_fold(
        X_tr_sc[tr_idx], y_sqrt[tr_idx],
        X_tr_sc[va_idx], y_sqrt[va_idx],
        epochs=200, patience=20, batch_size=64, lr=1e-3,
    )
    oof[va_idx] = np.clip(preds_sqrt, 0, None) ** 2
    sp_rmse = loso_rmse(oof[va_idx], y_train[va_idx])
    print(f"  Fold {fold_i+1:2d} (species={sp:2d})  RMSE={sp_rmse:.2f}")

rmse_x = loso_rmse(oof, y_train)
print()
print(f"=== RESULT ===")
print(f"[X] 1D-CNN  LOSO-RMSE = {rmse_x:.4f}")
print(f"    vs T(19.55): {rmse_x - 19.55:+.4f}")

# ── 残差プロット ──────────────────────────────────────────────────────────────
plot_residuals(oof, y_train, sp_train, EXP_LETTER,
               title=f"Exp X [1D-CNN]  LOSO={rmse_x:.4f}")

# ── 最終モデル訓練・提出ファイル生成 ──────────────────────────────────────────
final_model = train_final(X_tr_sc, y_sqrt, n_rounds=150)
final_model.eval()
with torch.no_grad():
    preds_sqrt = final_model(torch.tensor(X_te_sc)).numpy()
preds = np.clip(preds_sqrt, 0, None) ** 2
save_submission(test_ids, preds, OUT_PATH)
