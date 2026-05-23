"""
Experiment S3: di-PLS (Domain-Invariant PLS)
=============================================
仮説:
  EPOは訓練樹種間の変動方向しか除去できない。
  di-PLSはテスト樹種のX分布を直接活用してドメインシフトを補正する。
  NIRケモメトリクスでRMSEP 46〜80%削減の報告例あり。

LOSO整合性:
  fold i (樹種 s_i を保留):
    X_src = X_train[sp != s_i]  (12樹種, ラベルあり)
    X_tgt = X_train[sp == s_i]  (1樹種, ラベルなし扱い)
    m.fit(X_src, y_src, xs=X_src, xt=X_tgt)
    oof[sp == s_i] = m.predict(X_tgt)

  最終予測:
    m_final.fit(X_train, y_train, xs=X_train, xt=X_test)
    → テスト6樹種のX分布を直接活用

ルール整合性:
  テストXはラベルなしで使用 = transductive learning = 正当。
  テストyは使用しない。

前処理: MSC+SG(w=9,p=2) (EPO不要 — di-PLSがドメインシフトを直接補正)
グリッド: A=[5,10,15,20,30], l=[0.01,0.1,1,10,100]
さらに: y^0.27変換との組み合わせも探索

ベース: P1 (LOSO=15.4725, LB=15.395)
期待改善: -1.0〜3.0
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate,
)
from diPLSlib.models import DIPLS
import warnings; warnings.filterwarnings("ignore")

EXP = "S3"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

P1_BASELINE = 15.4725
P1_LB       = 15.395

# ── Data & preprocessing ──────────────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr    = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte    = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)

print(f"=== Experiment {EXP}: di-PLS (Domain-Invariant PLS) ===")
print(f"ベース: P1 (LOSO={P1_BASELINE}, LB={P1_LB})")
print(f"前処理: MSC+SG(w=9,p=2)  EPOなし(di-PLSがドメインシフト補正)\n")


def run_dipls_loso(Xtr, y, sp, A, l, y_power=1.0):
    """
    di-PLS LOSO-CV。
    各foldで保留樹種のXをtarget domainとして渡す。
    y_power: target transformation (1.0=無変換, 0.27など)
    """
    y_t = y ** y_power if y_power != 1.0 else y
    oof = np.zeros(len(y))

    for tr_idx, va_idx, _ in loso_folds(sp):
        X_src = Xtr[tr_idx]
        y_src = y_t[tr_idx].reshape(-1, 1)
        X_tgt = Xtr[va_idx]  # ラベルなし扱い (target domain)

        try:
            m = DIPLS(A=A, l=l)
            m.fit(X_src, y_src, xs=X_src, xt=X_tgt)
            pred = m.predict(X_tgt)
        except Exception as e:
            # di-PLSが収束しない場合はPLS単体にフォールバック
            m = DIPLS(A=A, l=0)
            m.fit(X_src, y_src, xs=X_src, xt=X_tgt)
            pred = m.predict(X_tgt)

        if y_power != 1.0:
            pred = np.clip(pred, 0, None) ** (1.0 / y_power)
        oof[va_idx] = pred

    return loso_rmse(oof, y), oof


# ── Phase 1: y変換なし、グリッドサーチ ────────────────────────────────────
print("Phase 1: y変換なし")
print(f"{'A':>4}  {'l':>6}  {'LOSO':>8}  {'vs P1':>7}")
print("-" * 35)

best_rmse = np.inf; best_A = None; best_l = None

for A in [5, 10, 15, 20]:
    for l in [0.01, 0.1, 1.0, 10.0, 100.0]:
        try:
            rmse, _ = run_dipls_loso(Xtr, y_train, sp_train, A=A, l=l, y_power=1.0)
        except Exception as e:
            print(f"  A={A:2d}  l={l:6.2f}  ERROR: {e}")
            continue
        diff = rmse - P1_BASELINE
        flag = " <-- best" if rmse < best_rmse else ""
        print(f"  A={A:2d}  l={l:6.2f}  {rmse:8.4f}  {diff:+7.4f}{flag}")
        if rmse < best_rmse:
            best_rmse = rmse; best_A = A; best_l = l

print(f"\nPhase 1 Best: A={best_A}, l={best_l}  LOSO={best_rmse:.4f}")

# ── Phase 2: y^0.27変換あり ────────────────────────────────────────────────
print(f"\nPhase 2: y^0.27変換あり (best A={best_A}前後を探索)")
print(f"{'A':>4}  {'l':>6}  {'LOSO':>8}  {'vs P1':>7}")
print("-" * 35)

best_rmse2 = np.inf; best_A2 = None; best_l2 = None; best_power = 1.0

for A in [max(3, best_A-5), best_A, best_A+5, best_A+10]:
    if A <= 0 or A > 40: continue
    for l in [best_l / 10, best_l, best_l * 10]:
        try:
            rmse, _ = run_dipls_loso(Xtr, y_train, sp_train, A=A, l=l, y_power=0.27)
        except Exception as e:
            print(f"  A={A:2d}  l={l:6.2f}  ERROR: {e}")
            continue
        diff = rmse - P1_BASELINE
        flag = " <-- best" if rmse < best_rmse2 else ""
        print(f"  A={A:2d}  l={l:6.2f}  {rmse:8.4f}  {diff:+7.4f}{flag}")
        if rmse < best_rmse2:
            best_rmse2 = rmse; best_A2 = A; best_l2 = l; best_power = 0.27

# 全体ベスト
if best_rmse2 < best_rmse:
    overall_best_rmse = best_rmse2
    overall_A, overall_l, overall_power = best_A2, best_l2, 0.27
    print(f"\n→ y^0.27変換ありが勝利")
else:
    overall_best_rmse = best_rmse
    overall_A, overall_l, overall_power = best_A, best_l, 1.0
    print(f"\n→ y変換なしが勝利")

print(f"\n=== Overall Best: A={overall_A}, l={overall_l}, power={overall_power} ===")
print(f"    LOSO={overall_best_rmse:.4f}  vs P1: {overall_best_rmse-P1_BASELINE:+.4f}")

# ── 最終モデル構築・提出 ──────────────────────────────────────────────────────
y_final = y_train ** overall_power if overall_power != 1.0 else y_train
m_final = DIPLS(A=overall_A, l=overall_l)
m_final.fit(X_train_raw if False else Xtr,   # preprocessed train
            y_final.reshape(-1, 1),
            xs=Xtr,
            xt=Xte)   # ← テスト6樹種のX分布を活用
preds_raw = m_final.predict(Xte)
preds = np.clip(preds_raw, 0, None) ** (1.0/overall_power) if overall_power != 1.0 else np.clip(preds_raw, 0, None)

power_tag = f"p{int(overall_power*100):03d}" if overall_power != 1.0 else "nop"
OUT = f"{OUT_DIR}/submission_{EXP}_A{overall_A}_l{int(overall_l*100)}_{power_tag}.csv"
save_submission(test_ids, preds, OUT)

memo = f"{EXP}: di-PLS(A={overall_A},l={overall_l},pow={overall_power}) LOSO={overall_best_rmse:.4f}"
submit_to_signate(OUT, memo, loso=overall_best_rmse)
