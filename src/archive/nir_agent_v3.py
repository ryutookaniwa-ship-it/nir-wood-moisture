"""
NIR DS Agent v3 — LangGraph + Gemini API
==========================================
v2からの更新:
  - initial_history を Q1-U3 まで全実験反映
  - EXP_LETTERS を V1〜 系列に更新
  - プランナープロンプトを最新知見に全面刷新
  - LOSO提出閾値を 15.1 に引き下げ（LBベスト15.395に対応）

Loop: planner -> coder -> executor -> [error: coder | success: analyzer] -> planner

現状:
  LBベスト: P1 = 15.395 (LOSO=15.4725, gap=-0.077)
  パイプライン: MSC → SG(w=9,p=2) → EPO(n=5) → y^0.27 → LGBM(lr=0.02,leaves=63,ff=0.07,mcs=10)
  ボトルネック: sp15(RMSE≈39, MC_max=298%) が全体を+4.4 押し上げ
               sp11(RMSE≈20, bias=+11) が第2ボトルネック

Run:
  python src/nir/nir_agent_v3.py              # auto-loop (max 5 iterations)
  python src/nir/nir_agent_v3.py --once       # 1 iteration
  python src/nir/nir_agent_v3.py --iters 3   # N iterations

Requires: GEMINI_API_KEY environment variable

Rate limits (Gemini free tier):
  gemini-2.5-pro:   5 RPM, 25 RPD
  gemini-2.0-flash: 15 RPM, 1500 RPD
"""

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import TypedDict, Optional

from dotenv import load_dotenv
load_dotenv(r"C:/Users/ryuch/OneDrive/デスクトップ/my_kaggle_project/.env")

from google import genai
from google.genai import types as genai_types
from langgraph.graph import StateGraph, END

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("C:/Users/ryuch/OneDrive/デスクトップ/my_kaggle_project")
PYTHON     = sys.executable
SCORES_MD  = BASE_DIR / "scores" / "nir-wood-moisture" / "scores.md"
SRC_DIR    = BASE_DIR / "src"
OUT_DIR    = BASE_DIR / "output" / "nir-wood-moisture"
PLOTS_DIR  = BASE_DIR / "output" / "agent_plots"

# 実験レター管理: V1〜 系列 (A-U系は既に使用済み)
EXP_LETTERS = (
    [f"V{i}" for i in range(1, 20)]
    + [f"W{i}" for i in range(1, 20)]
    + [f"X{i}" for i in range(1, 20)]
)

MAX_RETRIES = 2
MAX_ITERS   = 5
LOSO_SUBMIT_THRESHOLD = 15.1   # LBベスト15.395の -0.3 以上改善時のみ提出

# ── Gemini クライアント & レート制限 ─────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PLANNER_MODEL  = "gemini-2.0-flash"
CODER_MODEL    = "gemini-2.0-flash"
ANALYZER_MODEL = "gemini-2.0-flash"

MIN_CALL_INTERVAL = 8    # 秒 (15RPM → 4s + 余裕。TPM対策で余裕を増やす)
_last_call_time: dict[str, float] = {}


# ── State ────────────────────────────────────────────────────────────────────
class DSState(TypedDict):
    iteration:   int
    exp_letter:  str
    hypothesis:  str
    script_path: str
    code:        str
    logs:        str
    loso_score:  Optional[float]
    error:       Optional[str]
    retry_count: int
    history:     list


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_scores(last_n_rows: int = 20) -> str:
    """scores.md を読んで打ち手ログ表の最新N行だけ返す（トークン節約）。"""
    content = SCORES_MD.read_text(encoding="utf-8")
    lines = content.split("\n")
    table_rows = [l for l in lines if l.strip().startswith("|") and "---" not in l]
    header = table_rows[0] if table_rows else ""
    recent = table_rows[max(1, len(table_rows) - last_n_rows):]
    return f"Recent experiments (last {last_n_rows}):\n{header}\n" + "\n".join(recent)


def update_scores_md(letter: str, hypothesis: str, loso: float | None,
                     submitted: bool, analysis: str) -> None:
    """Append agent experiment row to scores.md 打ち手ログ table."""
    content = SCORES_MD.read_text(encoding="utf-8")
    lines = content.split("\n")

    loso_str       = f"**{loso:.4f}**" if loso else "—"
    lb_str         = "提出済" if submitted else "—"
    baseline_loso  = 15.4725  # P1 LOSO
    delta_str      = f"{loso - baseline_loso:+.4f}" if loso else "—"
    hyp_short      = hypothesis[:55].replace("|", "/")
    analysis_short = analysis[:80].replace("|", "/").replace("\n", " ")
    new_row = (f"| **{letter}** | {hyp_short} | LGBM(P1-params) | — "
               f"| {loso_str} | {lb_str} | Agent delta={delta_str}. {analysis_short} |")

    last_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and "---" not in stripped:
            last_idx = i
    if last_idx >= 0:
        lines.insert(last_idx + 1, new_row)
        SCORES_MD.write_text("\n".join(lines), encoding="utf-8")
        print(f"[SCORES] scores.md updated: {letter}  LOSO={loso}")
    else:
        print("[SCORES] WARNING: table not found in scores.md")


def gemini(system: str, user: str, model: str, max_tokens: int = 4096) -> str:
    """Gemini API呼び出し。レート制限 + 指数バックオフ付き。"""
    last = _last_call_time.get(model, 0.0)
    wait = MIN_CALL_INTERVAL - (time.time() - last)
    if wait > 0:
        print(f"[API] Rate limit: waiting {wait:.1f}s for {model} ...")
        time.sleep(wait)

    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"{system}\n\n{user}",
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
            )
            _last_call_time[model] = time.time()
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                backoff = 60 * (2 ** attempt)  # 60, 120, 240, 480, 960, 1920s
                print(f"[API] 429 Rate limit hit ({model}). Backing off {backoff}s ({backoff//60}min) ...")
                time.sleep(backoff)
            else:
                raise
    raise RuntimeError(f"Gemini API failed after retries: {model}")


def extract_loso(text: str) -> Optional[float]:
    for pat in [
        r"LOSO[-_\s]?RMSE\s*[:=]\s*([\d.]+)",
        r"LOSO\s*=\s*([\d.]+)",
        r"loso\s*=\s*([\d.]+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def next_exp_letter(history: list) -> str:
    used = {h["letter"] for h in history}
    for letter in EXP_LETTERS:
        if letter not in used:
            return letter
    return "Z99"


# ── Code template ─────────────────────────────────────────────────────────────
CODE_TEMPLATE = '''\
"""Experiment {letter}: {title}"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
sys.path.insert(0, r"{src_dir}")
sys.path.insert(0, r"{src_dir}/nir")

from nir_loso_utils import (
    load_data, msc, snv, sg_deriv,
    loso_folds, loso_rmse, loso_lgbm, loso_sklearn,
    save_submission, submit_to_signate, LGBM_BASE_PARAMS,
)
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

EXP    = "{letter}"
OUT    = r"{out_path}"

# ── P1ベースパイプライン (MSC+SG(w=9,p=2)+EPO(n=5)+y^0.27) ─────────────────
def compute_epo_matrix(X, y, sp, bin_width=10.0, n_components=5, min_species=2):
    bins = np.arange(0, y.max() + bin_width, bin_width)
    all_dirs = []
    for lo in bins[:-1]:
        hi = lo + bin_width; mask = (y >= lo) & (y < hi)
        if mask.sum() < 4: continue
        sp_in = np.unique(sp[mask])
        if len(sp_in) < min_species: continue
        sp_means = np.array([X[mask][sp[mask]==s].mean(axis=0) for s in sp_in])
        inter = sp_means - sp_means.mean(axis=0)
        n_c = min(n_components, inter.shape[0]-1)
        if n_c < 1: continue
        pca = PCA(n_components=n_c, random_state=42); pca.fit(inter)
        all_dirs.append(pca.components_)
    if not all_dirs: return np.zeros((X.shape[1], 1))
    D = np.vstack(all_dirs); _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:n_components].T

def apply_epo(X, V): return X - (X @ V) @ V.T

# P1固定パラメータ (LBベスト: 15.395)
P1_PARAMS = {{**LGBM_BASE_PARAMS,
              "learning_rate": 0.02, "num_leaves": 63,
              "feature_fraction": 0.07, "min_child_samples": 10}}

data = load_data()
y_train     = data["y_train"]
X_train_raw = data["X_train_raw"]
X_test_raw  = data["X_test_raw"]
test_ids    = data["test_ids"]
sp_train    = data["sp_train"]

ref    = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

# P1のデフォルトターゲット変換
y_sqrt = y_train ** 0.27

# ===== EXPERIMENT CODE BELOW =====
# 利用可能な変数:
#   Xtr (1322x1555, EPO適用済み), Xte (550x1555, EPO適用済み)
#   y_train (生MC%), y_sqrt (y^0.27変換済み), sp_train, test_ids
#   P1_PARAMS, LGBM_BASE_PARAMS
#
# 必須出力:
#   print(f"LOSO-RMSE = {{rmse:.4f}}")  ← この形式必須
#   save_submission(test_ids, preds, OUT)
#   submit_to_signate(OUT, f"Exp{EXP}: ... LOSO={{rmse:.4f}}", loso=rmse)
#     (LOSO >= {threshold} なら自動スキップ)
#
# LOSOループ例:
#   for tr_idx, va_idx, sp in loso_folds(sp_train):
#       ...
# ===== END INSTRUCTIONS =====
'''


# ── Node: planner ─────────────────────────────────────────────────────────────
def planner(state: DSState) -> DSState:
    print(f"\n{'='*60}")
    print(f"[PLANNER] Iteration {state['iteration']+1}/{MAX_ITERS}")
    print(f"{'='*60}")

    scores_text = read_scores()
    history_text = ""
    # 直近5件のみ渡す（トークン節約）
    recent_history = state["history"][-5:] if state["history"] else []
    if recent_history:
        history_text = "\n\nRecent agent history (last 5):\n"
        for h in recent_history:
            history_text += (
                f"  {h['letter']}: LOSO={h.get('loso_score','?')} delta={h.get('loso_delta','?')} "
                f"— {h['hypothesis'][:60]}\n"
            )

    system = (
        "You are an expert NIR spectroscopy data scientist competing on SIGNATE. "
        "Goal: minimize LOSO-RMSE (Leave-One-Species-Out CV) for wood moisture prediction from NIR spectra.\n\n"

        "=== CURRENT STATE ===\n"
        "LB best: P1 = 15.395 (LOSO=15.4725, gap=-0.077)\n"
        "Pipeline: MSC → SG(w=9,p=2,deriv=1) → EPO(n=5) → y^0.27 → LGBM(lr=0.02, leaves=63, ff=0.07, mcs=10)\n"
        "Auto-submit threshold: LOSO < 15.1 only\n\n"

        "=== ROOT CAUSE DIAGNOSIS ===\n"
        "1. sp15 (species 15): RMSE≈39, bias=-6.8, n=112, MC_max=298.6%\n"
        "   → Free water domain (MC >> FSP=30%). Spectra differ structurally from all train species.\n"
        "   → Removing sp15 gives LOSO=11.07 (vs full LOSO=15.47), confirming sp15 adds +4.40.\n"
        "   → LB=15.40 >> ex-sp15 LOSO=11.07, meaning test species are harder than train species.\n"
        "2. sp11 (species 11): RMSE≈20, bias=+11 (overprediction)\n"
        "3. High MC domain (>150%): catastrophic underprediction by current model\n"
        "4. EPO (n=5) successfully removes inter-species spectral variation (confirmed: test-train distance ≈ 0 post-EPO)\n\n"

        "=== CONFIRMED OPTIMAL (do NOT change these) ===\n"
        "- MSC scatter correction (better than SNV; Raw=22.77 is catastrophically worse)\n"
        "- SG window=9, poly=2 (J4 fine-grained search confirms this)\n"
        "- EPO n=5 global fit (J1 confirmed; n=6 rapidly worse; fold-internal EPO=LOSO+5.18)\n"
        "- feature_fraction=0.07 (I1/J2 confirmed: sweet spot regardless of SG config)\n"
        "- lr=0.02, leaves=63, mcs=10 (L1 confirmed: leaves=31+mcs=30 overfits train species)\n"
        "- Target transform: y^0.27 (P6 super-fine-grained: 0.270 is true optimal)\n"
        "- Loss: L2 regression (M1/S2: Huber/MAE/Tweedie all worse)\n\n"

        "=== EXHAUSTIVELY FAILED APPROACHES (do NOT repeat any) ===\n"
        "Preprocessing:\n"
        "  - EMSC(poly=1): LB=20.05 (CV optimistic bias +2.95, test species harmed)\n"
        "  - Multi-scale SG [5,9,13]: +0.60 worse; ff=0.07 insufficient coverage of 4665 dims\n"
        "  - 2-scale SG [5,13]: +0.82; mixed-poly (w5p3+w9p2): +0.86\n"
        "  - Species centring: +4.52 (sp mean contains moisture signal)\n"
        "  - Test-spectrum EPO: +10 to +13 catastrophic (test PCA directions contain moisture signal)\n"
        "  - OSC: severe LOSO leakage (honest LOSO=52 vs leaky=6.4)\n"
        "  - Post-EPO normalization (SNV/MSC/StdScaler): all worse\n"
        "  - PCA compression after EPO: G2 LOSO=21.20 (ff=0.07 effect nullified)\n"
        "\nModels:\n"
        "  - MLP: LOSO improves → LB worsens (inverse correlation confirmed; F1 gap=+9.69)\n"
        "  - 1D Transformer: gap=+3.13\n"
        "  - 1D CNN: +7.23; RandomForest: +0.95; XGBoost: +0.66\n"
        "  - PLS after EPO: LB=26.37 (re-learns species patterns despite EPO)\n"
        "  - CatBoost: sp15 RMSE=61, worse\n"
        "  - SVR+EPO: +6.15; ExtraTrees: +1.13\n"
        "\nEnsemble/Post-processing:\n"
        "  - Seed ensemble (5 seeds): r=0.998, diversity ≈ 0\n"
        "  - LOSO bagging (13 species models): r=0.9963, diversity ≈ 0\n"
        "  - I2×B2 ensemble: LB worse despite LOSO improvement\n"
        "  - Stacking (4 preprocessing variants): r=0.99, diversity ≈ 0\n"
        "\nTarget / Loss:\n"
        "  - DART booster: 800 rounds insufficient; +7.49\n"
        "  - Tweedie (all vp): worse than L2\n"
        "  - Huber+p=0.27: identical to L2 in y^0.27 space\n"
        "  - Isotonic calibration: LOSO better → LB +2.24\n"
        "  - Linear/poly calibration on all samples or sp15-only: LOSO better → LB worse\n"
        "  - Two-stage model (global + sp15 fix): LOSO 14.78 → LB +1.30 (sp15 over-correction)\n"
        "  - sample_weight(MC>100%, w=2.0) + p=0.27: worse than P1\n"
        "\nFeature engineering:\n"
        "  - Water band masking (5187+6896+8333cm⁻¹, 310 pts): +2.23 worse\n"
        "  - Physical features (water band area/ratio): no improvement (EPO already handles)\n"
        "  - PCA expansion of sp15 PCA directions (k=2,3,5): removes moisture signal\n"
        "\nParameter search (all confirmed at P1 values):\n"
        "  - EPO n_comp (2-10): n=5 optimal; n≥6 rapidly worse\n"
        "  - EPO bin_width (5-30): bw=10 optimal\n"
        "  - feature_fraction (0.02-0.15): 0.07 confirmed\n"
        "  - lr (0.005-0.02): 0.02 best\n"
        "  - leaves/mcs retuning with p=0.27: Q2 LOSO improved but LB worsened\n"
        "  - Target power p (0.255-0.290): p=0.270 confirmed true optimal\n\n"

        "=== PROMISING UNEXPLORED DIRECTIONS ===\n"
        "Consider these carefully — they are NOT confirmed to fail:\n"
        "1. **Quantile regression or interval-based loss**: Instead of minimizing MSE in y^0.27 "
        "   space, try minimizing a custom asymmetric loss that penalizes underprediction more "
        "   (since sp15 shows systematic underprediction at high MC). "
        "   Note: Huber failed, but asymmetric pinball/quantile loss is different.\n"
        "2. **Spectral ratio features**: After EPO, compute ratios of key wavenumber pairs "
        "   (e.g., absorbance at 5187 / absorbance at 5900) as additional features. "
        "   Ratios can be species-invariant while encoding moisture.\n"
        "3. **Stacking with diverse preprocessing seeds**: Run EPO with different random_state "
        "   values to get weakly diverse EPO matrices, then average predictions.\n"
        "4. **Gradient boosting with max_bin tuning**: The current LGBM uses default max_bin=255. "
        "   Higher max_bin (511, 1023) might improve resolution for high-MC samples.\n"
        "5. **Monotonic constraints**: Moisture content should monotonically increase with "
        "   absorption at 5187/6896 cm⁻¹. Adding monotone_constraints to LGBM.\n"
        "6. **Bagging with different EPO seeds**: EPO with different random_state for PCA "
        "   produces slightly different projection matrices → genuine diversity.\n\n"

        "Respond ONLY with JSON: {\"hypothesis\": \"...\", \"title\": \"...\", \"exp_type\": \"...\"}"
    )

    user_msg = (
        f"Score tracker:\n{scores_text}"
        f"{history_text}\n\n"
        "Propose the single most promising next experiment that has NOT been tried. "
        "Focus on the sp15/high-MC underprediction problem. "
        "Be specific about the exact implementation. "
        "Avoid ALL approaches listed in EXHAUSTIVELY FAILED. "
        "Respond ONLY with JSON."
    )

    raw = gemini(system, user_msg, model=PLANNER_MODEL, max_tokens=1024)
    raw_clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
    hypothesis = ""; title = "Agent Experiment"
    m = re.search(r'\{.*\}', raw_clean, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            hypothesis = d.get("hypothesis", "")
            title      = d.get("title", "Agent Experiment")
        except json.JSONDecodeError:
            pass
    if not hypothesis:
        hm = re.search(r'"hypothesis"\s*:\s*"(.*?)"(?:,|\s*\})', raw_clean, re.DOTALL)
        hypothesis = hm.group(1).replace('\\"', '"') if hm else raw_clean[:300]
    if not title or title == "Agent Experiment":
        tm = re.search(r'"title"\s*:\s*"(.*?)"', raw_clean)
        if tm: title = tm.group(1)

    letter = next_exp_letter(state["history"])
    print(f"[PLANNER] Exp {letter}: {title}")
    print(f"[PLANNER] Hypothesis: {hypothesis}")

    return {**state, "exp_letter": letter, "hypothesis": hypothesis,
            "error": None, "retry_count": 0}


# ── Node: coder ───────────────────────────────────────────────────────────────
def coder(state: DSState) -> DSState:
    letter = state["exp_letter"]
    hypo   = state["hypothesis"]
    error  = state.get("error", "")
    retry  = state.get("retry_count", 0)

    print(f"\n[CODER] Generating script for Exp {letter} (retry={retry})")

    out_path = str(OUT_DIR / f"submission_{letter}_agent.csv")
    template = CODE_TEMPLATE.format(
        letter=letter,
        title=hypo[:60],
        src_dir=str(SRC_DIR),
        out_path=out_path,
        threshold=LOSO_SUBMIT_THRESHOLD,
    )

    error_ctx = ""
    if error and retry > 0:
        error_ctx = f"\n\nPrevious error:\n{error[:800]}\nPlease fix."

    system = (
        "You are an expert Python data scientist. Generate a complete runnable script "
        "for a NIR spectroscopy experiment.\n\n"
        "AVAILABLE FUNCTIONS (exact signatures from nir_loso_utils):\n"
        "  msc(X, reference=None) -> ndarray\n"
        "  snv(X) -> ndarray\n"
        "  sg_deriv(X, window=11, polyorder=2, deriv=1) -> ndarray\n"
        "  loso_folds(sp_train) -> yields (tr_idx, va_idx, sp)\n"
        "  loso_rmse(oof, y) -> float\n"
        "  loso_lgbm(X_tr, y, sp_train, params, n_rounds=1000) -> (rmse, avg_rounds, oof)\n"
        "  save_submission(test_ids, preds, path) -> None\n"
        "  submit_to_signate(path, memo, loso=float) -> None\n"
        "  LGBM_BASE_PARAMS, P1_PARAMS (already defined)\n"
        "  Xtr, Xte, y_train, y_sqrt, sp_train, test_ids (already defined)\n\n"
        "RULES:\n"
        "1. DO NOT redefine: msc, snv, sg_deriv, loso_folds, loso_rmse, loso_lgbm, "
        "   compute_epo_matrix, apply_epo, Xtr, Xte, y_train, y_sqrt, sp_train, test_ids\n"
        "2. MUST print: print(f'LOSO-RMSE = {rmse:.4f}')\n"
        "3. MUST call: save_submission(test_ids, preds, OUT)\n"
        "4. MUST call: submit_to_signate(OUT, f'ExpXX: ... LOSO={rmse:.4f}', loso=rmse)\n"
        "5. For custom objectives use lgb.train directly (not loso_lgbm wrapper)\n"
        "6. When predicting, inverse-transform if you changed the target: "
        "   e.g. preds_raw = model.predict(Xte); preds = preds_raw ** (1/0.27)\n"
        "7. Return ONLY Python code, no markdown fences.\n\n"
        f"Template:\n{template}"
    )

    user_msg = (
        f"Implement: {hypo}{error_ctx}\n\n"
        "Fill in the EXPERIMENT CODE section only. "
        "Xtr/Xte are MSC+SG+EPO preprocessed (1555 features). "
        "y_sqrt = y_train**0.27 is the default target. "
        "Print 'LOSO-RMSE = X.XXXX'."
    )

    code = gemini(system, user_msg, model=CODER_MODEL, max_tokens=4096)
    code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()

    script_path = str(SRC_DIR / "nir" / f"nir_agent_exp_{letter}.py")
    Path(script_path).write_text(code, encoding="utf-8")
    print(f"[CODER] Script saved: {script_path}")

    new_retry = state["retry_count"] + 1 if state.get("error") else state["retry_count"]
    return {**state, "code": code, "script_path": script_path, "retry_count": new_retry}


# ── Node: executor ────────────────────────────────────────────────────────────
def executor(state: DSState) -> DSState:
    script = state["script_path"]
    print(f"\n[EXECUTOR] Running: {Path(script).name}")

    result = subprocess.run(
        [PYTHON, script],
        capture_output=True, text=True,
        timeout=900, cwd=str(BASE_DIR),
    )

    logs = result.stdout
    if result.returncode != 0:
        logs += f"\n\nSTDERR:\n{result.stderr}"

    print(f"[EXECUTOR] Return code: {result.returncode}")
    for line in logs.splitlines()[-15:]:
        print(f"  {line}")

    loso_score = extract_loso(logs)
    error = None if result.returncode == 0 else result.stderr[:2000]

    if loso_score:
        print(f"[EXECUTOR] LOSO-RMSE: {loso_score:.4f}  (threshold={LOSO_SUBMIT_THRESHOLD})")
    if error:
        print(f"[EXECUTOR] ERROR detected")

    return {**state, "logs": logs, "loso_score": loso_score, "error": error}


# ── Node: analyzer ────────────────────────────────────────────────────────────
def analyzer(state: DSState) -> DSState:
    letter = state["exp_letter"]
    print(f"\n[ANALYZER] Exp {letter}")

    system = (
        "You are an expert NIR spectroscopy data scientist. "
        "Concisely analyze experiment results (3-5 sentences): "
        "(1) LOSO vs P1 baseline (15.4725)? "
        "(2) What does this tell us about sp15/high-MC extrapolation? "
        "(3) What specific next step does this suggest?"
    )
    user_msg = (
        f"Exp {letter}: {state['hypothesis']}\n"
        f"LOSO-RMSE: {state['loso_score']} (P1 baseline=15.4725, LB best=15.395)\n\n"
        f"Output:\n{state['logs'][-2000:]}"
    )

    analysis = gemini(system, user_msg, model=ANALYZER_MODEL, max_tokens=512)
    print(f"[ANALYZER] {analysis[:300]}")

    baseline = 15.4725
    loso = state["loso_score"]
    delta = (loso - baseline) if loso else None
    submitted = loso is not None and loso < LOSO_SUBMIT_THRESHOLD
    history_entry = {
        "letter":      letter,
        "hypothesis":  state["hypothesis"],
        "loso_score":  loso,
        "loso_delta":  f"{delta:+.4f}" if delta is not None else "N/A",
        "analysis":    analysis,
    }

    update_scores_md(letter, state["hypothesis"], loso, submitted, analysis)

    return {
        **state,
        "analysis":  analysis,
        "history":   state["history"] + [history_entry],
        "iteration": state["iteration"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_executor(state: DSState) -> str:
    if state.get("error") and state["retry_count"] < MAX_RETRIES:
        print(f"[ROUTER] Error → retry {state['retry_count']+1}/{MAX_RETRIES}")
        return "coder"
    return "analyzer"


def route_after_analyzer(state: DSState) -> str:
    if state["iteration"] >= MAX_ITERS:
        print(f"[ROUTER] Max iterations ({MAX_ITERS}) reached.")
        return END
    return "planner"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(DSState)
    graph.add_node("planner",  planner)
    graph.add_node("coder",    coder)
    graph.add_node("executor", executor)
    graph.add_node("analyzer", analyzer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder",   "executor")
    graph.add_conditional_edges("executor", route_after_executor,
                                 {"coder": "coder", "analyzer": "analyzer"})
    graph.add_conditional_edges("analyzer", route_after_analyzer,
                                 {"planner": "planner", END: END})
    return graph.compile()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--once" in args:
        max_iters = 1
    elif "--iters" in args:
        idx = args.index("--iters")
        max_iters = int(args[idx + 1])
    else:
        max_iters = MAX_ITERS

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    # 手動実験の全履歴 (v3ではQ1-U3まで反映)
    initial_history = [
        # ── ターゲット変換系 ──────────────────────────────────────────────
        {"letter": "M2",  "hypothesis": "べき乗変換p=0.30 → LB=15.545",
         "loso_score": 15.5877, "loso_delta": "+0.1152",
         "analysis": "p=0.30でLOSO=15.5877。高MC域圧縮が有効。"},
        {"letter": "P1",  "hypothesis": "細粒度p探索でp=0.27が最良 → LB=15.395(新ベスト)",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "P1=LBベスト。gap=-0.077(LBがLOSOより良い)。P6で0.270が真の最適確定。"},
        {"letter": "P2",  "hypothesis": "sample_weight(MC>100%, w=2.0) + p=0.30",
         "loso_score": 15.5347, "loso_delta": "+0.0622",
         "analysis": "M2比微改善のみ。p=0.27には及ばず。"},
        {"letter": "P3",  "hypothesis": "DART booster + p=0.30",
         "loso_score": 23.07,   "loso_delta": "+7.5975",
         "analysis": "800roundsで収束不足。大幅悪化。"},
        {"letter": "P4",  "hypothesis": "Tweedie目的関数(vp=1.2〜1.8)",
         "loso_score": 15.84,   "loso_delta": "+0.3675",
         "analysis": "全設定でM2より悪化。Tweedie不採用。"},
        {"letter": "P5",  "hypothesis": "p=0.27 + sample_weight(MC>100%, w=2.0) 組み合わせ",
         "loso_score": 15.5368, "loso_delta": "+0.0643",
         "analysis": "P1より悪化。重み付けとp=0.27は相乗効果なし。"},
        {"letter": "P6",  "hypothesis": "p=0.270周辺超細粒度探索(0.255〜0.290)",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "p=0.270が真の最適と確定。0.265/0.278は悪化。変換チューニングの限界。"},
        # ── EPOパラメータ系 ────────────────────────────────────────────────
        {"letter": "Q1",  "hypothesis": "EPO bin_width探索(5/7/10/15/20/30)",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "bw=10が依然最適と確定。小/大ビン共に悪化。"},
        {"letter": "Q2",  "hypothesis": "leaves/mcs再チューニング(p=0.27) → LB=15.535",
         "loso_score": 15.2130, "loso_delta": "-0.2595",
         "analysis": "LOSO-0.26改善もLB+0.14悪化。leaves=47,mcs=30が訓練種過適合。P1パラメータ確定。"},
        {"letter": "Q3",  "hypothesis": "EPO n_comp再探索(2〜8, p=0.27)",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "n=5が依然最適。n=6は+0.019、n=7で+2.37急激悪化。EPO全パラメータ確定。"},
        # ── 2段階モデル系 ──────────────────────────────────────────────────
        {"letter": "R1",  "hypothesis": "EPOなし(MSC+SG+LGBM) Q2パラメータ",
         "loso_score": 21.054,  "loso_delta": "+5.5815",
         "analysis": "EPOなしでsp15=58.70と更に悪化。EPOは必須。"},
        {"letter": "R2",  "hypothesis": "2段モデル(t=150%) + 重み(w=3, MC>100%) → LB=16.691",
         "loso_score": 14.776,  "loso_delta": "-0.6965",
         "analysis": "LOSO大幅改善もLB+1.30悪化。gap=+1.915。高MC重み付けがテスト樹種で逆効果。"},
        # ── sp15診断・補正系 ──────────────────────────────────────────────
        {"letter": "S1",  "hypothesis": "extra_trees=True + leaves探索",
         "loso_score": 16.6019, "loso_delta": "+1.1294",
         "analysis": "分割閾値ランダム化は逆効果。+1.13悪化。"},
        {"letter": "S2",  "hypothesis": "Huber損失(delta=0.5〜8.0) + p=0.27",
         "loso_score": 15.7004, "loso_delta": "+0.2279",
         "analysis": "全delta値で同一結果(15.70)。y^0.27空間でHuber≈L2。"},
        {"letter": "S3",  "hypothesis": "di-PLS (A=5〜20, l=0〜1)",
         "loso_score": None,    "loso_delta": "N/A",
         "analysis": "sp1単折RMSE=33。lパラメータが完全無効。線形モデルの限界。"},
        {"letter": "S5",  "hypothesis": "Test-Spectrum EPO (k=1〜3)",
         "loso_score": 26.38,   "loso_delta": "+10.9075",
         "analysis": "壊滅的悪化(+10〜13)。テストXのPCA方向に含水率シグナル含有。絶対不可。"},
        {"letter": "S6",  "hypothesis": "LOSO Bagging(13モデル平均)",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "r=0.9963で多様性なし。効果ゼロ。"},
        {"letter": "S7",  "hypothesis": "OSC(k=5)+EPO → LGBM",
         "loso_score": 52.70,   "loso_delta": "+37.2275",  # honest LOSO
         "analysis": "LOSOリーク確定。honest LOSO=52.70 vs leaky=6.4。OSCはy使用→絶対不可。"},
        # ── マルチスケール・混合SG系 ───────────────────────────────────────
        {"letter": "T1",  "hypothesis": "マルチスケールSG w=[5,9,13] Joint-EPO ff=0.023",
         "loso_score": 16.0680, "loso_delta": "+0.5955",
         "analysis": "ff低すぎsp3収束せず。4665dim追加もP1比+0.60悪化。"},
        {"letter": "T1b", "hypothesis": "T1 ff grid(0.04〜0.07)",
         "loso_score": 15.7600, "loso_delta": "+0.2875",
         "analysis": "ff=0.07でもP1比+0.29。多スケール情報増分なし。"},
        {"letter": "T1c", "hypothesis": "2-scale SG w=[5,13] Joint-EPO",
         "loso_score": 16.2881, "loso_delta": "+0.8156",
         "analysis": "w=9なしで+0.82悪化。中周波除外は逆効果。"},
        {"letter": "T1d", "hypothesis": "混合poly (w5p3+w9p2) Joint-EPO",
         "loso_score": 16.3360, "loso_delta": "+0.8635",
         "analysis": "+0.86悪化。混合polyの情報増分なし。"},
        # ── sp15特化・キャリブレーション系 ──────────────────────────────
        {"letter": "U1",  "hypothesis": "P1診断 sp15除外LOSO分析",
         "loso_score": 15.4725, "loso_delta": "+0.0000",
         "analysis": "診断: sp15がLOSOを+4.40引き上げ。LB=15.40 >> ex-sp15 LOSO=11.07。"},
        {"letter": "U2",  "hypothesis": "EPO+sp15 within-PCA拡張(k=2,3,5)",
         "loso_score": 16.7748, "loso_delta": "+1.3023",
         "analysis": "sp15のPCA方向が水分シグナルも除去。全k値で悪化。"},
        {"letter": "U3",  "hypothesis": "線形・多項式キャリブレーション各種 → LB悪化",
         "loso_score": 14.6583, "loso_delta": "-0.8142",  # U3-A (best LOSO, worst LB)
         "analysis": "LOSO改善もLB+1〜+9悪化。訓練バイアスはテスト樹種に非適用。キャリブレーション不可。"},
    ]

    initial_state: DSState = {
        "iteration":   0,
        "exp_letter":  "V1",
        "hypothesis":  "",
        "script_path": "",
        "code":        "",
        "logs":        "",
        "loso_score":  None,
        "error":       None,
        "retry_count": 0,
        "history":     initial_history,
    }

    agent = build_agent()
    print("=" * 60)
    print(f"NIR DS Agent v3  |  max_iters={max_iters}")
    print(f"LB best: P1=15.395 (LOSO=15.4725, p=0.27)")
    print(f"Submit threshold: LOSO < {LOSO_SUBMIT_THRESHOLD}")
    print(f"Next exp: V1 (A-U fully exhausted)")
    print("=" * 60)

    final = agent.invoke(initial_state, config={"recursion_limit": 300})

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    for h in final.get("history", []):
        # Only show this session's experiments (V1 onwards)
        if h["letter"].startswith(("V", "W", "X")):
            submitted = "✓ SUBMITTED" if h.get("loso_score") and h["loso_score"] < LOSO_SUBMIT_THRESHOLD else ""
            print(f"  Exp {h['letter']}: LOSO={h.get('loso_score','?')} ({h.get('loso_delta','?')}) {submitted}")
            print(f"    {h['hypothesis'][:80]}")
    print("=" * 60)
