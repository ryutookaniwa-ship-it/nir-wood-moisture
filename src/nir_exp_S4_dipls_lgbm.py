"""
Experiment S4: di-PLS → LGBM (2段階: ドメイン不変潜在空間 + 非線形回収)
=========================================================================
仮説:
  di-PLS(S3)はドメイン不変な潜在空間(スコア行列T)を構築するが、
  最終回帰はPLS(線形)。
  そのスコア行列を入力としてLGBMを使えば非線形パターンも回収できる。

  E1(PLS→LGBM)失敗の理由は「通常PLSが樹種変動方向に沿って圧縮」。
  di-PLSはテスト樹種と整合した空間を構築するため根本的に異なる。

LOSO実装:
  fold i:
    m_dipls.fit(X_src, y_src, xs=X_src, xt=X_tgt)
    T_src = m_dipls.T_    # (n_src, A) ドメイン整合スコア
    T_tgt = m_dipls.Tt_   # (n_tgt, A) ターゲットスコア
    m_lgbm.fit(T_src, y_src)
    oof[tgt_idx] = m_lgbm.predict(T_tgt)

  最終モデル:
    m_dipls.fit(X_train, y_train, xt=X_test)
    T_train = m_dipls.T_
    T_test  = m_dipls.Tt_
    m_lgbm.fit(T_train, y_train)
    preds = m_lgbm.predict(T_test)

ベース: P1 (LOSO=15.4725, LB=15.395)
S3結果を踏まえてAとlを選択
期待改善: S3 -0.5〜1.0 追加
"""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src")
sys.path.insert(0, r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\src\nir")
from nir_loso_utils import (
    load_data, msc, sg_deriv, loso_folds, loso_rmse,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
from diPLSlib.models import DIPLS
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP = "S4"
OUT_DIR = r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project\output\nir-wood-moisture"

P1_BASELINE = 15.4725
P1_LB       = 15.395

# ── Data & preprocessing ──────────────────────────────────────────────────────
data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref = X_train_raw.mean(axis=0)
Xtr = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)

LGBM_PARAMS = {**LGBM_BASE_PARAMS,
               "learning_rate": 0.02, "num_leaves": 63,
               "feature_fraction": 0.5,   # 低次元空間(A成分)ではff=0.07は小さすぎ
               "min_child_samples": 5}

print(f"=== Experiment {EXP}: di-PLS → LGBM ===")
print(f"ベース: P1 (LOSO={P1_BASELINE}, LB={P1_LB})\n")


def run_dipls_lgbm_loso(Xtr, y, sp, A, l_dipls, y_power, ff=0.5, leaves=63):
    """
    LOSO-CV with di-PLS latent scores → LGBM.
    """
    y_t = y ** y_power if y_power != 1.0 else y
    oof = np.zeros(len(y))
    lgbm_p = {**LGBM_BASE_PARAMS,
              "learning_rate": 0.02, "num_leaves": leaves,
              "feature_fraction": ff, "min_child_samples": 5}

    for tr_idx, va_idx, _ in loso_folds(sp):
        X_src = Xtr[tr_idx]
        y_src = y_t[tr_idx].reshape(-1, 1)
        X_tgt = Xtr[va_idx]

        # di-PLS: ドメイン不変潜在空間を構築
        m_d = DIPLS(A=A, l=l_dipls)
        m_d.fit(X_src, y_src, xs=X_src, xt=X_tgt)

        T_src = m_d.T_   # (n_src, A) 訓練スコア
        T_tgt = m_d.Tt_  # (n_tgt, A) ターゲットスコア

        # LGBM: 低次元スコア空間で非線形回帰
        dtrain = lgb.Dataset(T_src, label=y_t[tr_idx])
        dval   = lgb.Dataset(T_tgt, label=y_t[va_idx], reference=dtrain)
        m_l = lgb.train(lgbm_p, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(-1)])
        pred = m_l.predict(T_tgt)
        if y_power != 1.0:
            pred = np.clip(pred, 0, None) ** (1.0 / y_power)
        oof[va_idx] = pred

    return loso_rmse(oof, y), oof


# ── Phase 1: y変換なし、AとlとLGBM-ffのグリッド ─────────────────────────────
print("Phase 1: y変換なし  (A, l_dipls, feature_fraction探索)")
print(f"{'A':>4}  {'l':>6}  {'ff':>5}  {'LOSO':>8}  {'vs P1':>7}")
print("-" * 45)

best_rmse = np.inf; best_cfg = None

# S3の結果をもとにAとlを絞る（未知なので広めに探索）
for A in [10, 15, 20]:
    for l in [0.1, 1.0, 10.0]:
        for ff in [0.3, 0.5, 0.7]:
            try:
                rmse, _ = run_dipls_lgbm_loso(Xtr, y_train, sp_train,
                                               A=A, l_dipls=l, y_power=1.0, ff=ff)
            except Exception as e:
                print(f"  A={A}  l={l}  ff={ff}  ERROR: {e}")
                continue
            diff = rmse - P1_BASELINE
            flag = " <-- best" if rmse < best_rmse else ""
            print(f"  A={A:2d}  l={l:6.2f}  ff={ff:.1f}  {rmse:8.4f}  {diff:+7.4f}{flag}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = dict(A=A, l=l, ff=ff, power=1.0)

# ── Phase 2: y^0.27変換あり ────────────────────────────────────────────────
if best_cfg:
    print(f"\nPhase 2: y^0.27変換 (best A={best_cfg['A']}, l={best_cfg['l']})")
    print(f"{'A':>4}  {'l':>6}  {'ff':>5}  {'LOSO':>8}  {'vs P1':>7}")
    print("-" * 45)

    for A in [best_cfg['A'] - 5, best_cfg['A'], best_cfg['A'] + 5]:
        if A <= 0: continue
        for l in [best_cfg['l'] / 10, best_cfg['l'], best_cfg['l'] * 10]:
            for ff in [best_cfg['ff']]:
                try:
                    rmse, _ = run_dipls_lgbm_loso(Xtr, y_train, sp_train,
                                                   A=A, l_dipls=l, y_power=0.27, ff=ff)
                except Exception as e:
                    print(f"  A={A}  l={l}  ff={ff}  ERROR: {e}")
                    continue
                diff = rmse - P1_BASELINE
                flag = " <-- best" if rmse < best_rmse else ""
                print(f"  A={A:2d}  l={l:6.2f}  ff={ff:.1f}  {rmse:8.4f}  {diff:+7.4f}{flag}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = dict(A=A, l=l, ff=ff, power=0.27)

print(f"\n=== Overall Best ===")
print(f"  A={best_cfg['A']}, l={best_cfg['l']}, ff={best_cfg['ff']}, "
      f"power={best_cfg['power']}")
print(f"  LOSO={best_rmse:.4f}  vs P1: {best_rmse-P1_BASELINE:+.4f}")

# ── 最終モデル構築・提出 ──────────────────────────────────────────────────────
A, l_d, ff_f, pw = best_cfg['A'], best_cfg['l'], best_cfg['ff'], best_cfg['power']
y_f = y_train ** pw if pw != 1.0 else y_train

m_final_d = DIPLS(A=A, l=l_d)
m_final_d.fit(Xtr, y_f.reshape(-1, 1), xs=Xtr, xt=Xte)
T_train = m_final_d.T_
T_test  = m_final_d.Tt_

lgbm_final_p = {**LGBM_BASE_PARAMS,
                "learning_rate": 0.02, "num_leaves": 63,
                "feature_fraction": ff_f, "min_child_samples": 5}
dtrain_f = lgb.Dataset(T_train, label=y_f)
# avg_iterを求めるために再度LOSOで推定（簡易: 500固定）
m_lgbm_f = lgb.train(lgbm_final_p, dtrain_f,
                      num_boost_round=500,
                      callbacks=[lgb.log_evaluation(-1)])

preds_raw = m_lgbm_f.predict(T_test)
preds = np.clip(preds_raw, 0, None) ** (1.0/pw) if pw != 1.0 else np.clip(preds_raw, 0, None)

power_tag = f"p{int(pw*100):03d}" if pw != 1.0 else "nop"
OUT = f"{OUT_DIR}/submission_{EXP}_A{A}_l{int(l_d*100)}_{power_tag}.csv"
save_submission(test_ids, preds, OUT)
memo = f"{EXP}: di-PLS(A={A},l={l_d})→LGBM(ff={ff_f}) LOSO={best_rmse:.4f}"
submit_to_signate(OUT, memo, loso=best_rmse)
