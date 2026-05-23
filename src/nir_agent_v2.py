"""
NIR DS Agent v2 — LangGraph + Gemini API
==========================================
Loop: planner -> coder -> executor -> [error: coder | success: analyzer] -> planner

現状:
  LBベスト: I2 (MSC+SG(w=9,p=2)+EPO(n=5)+LGBM) = LB 16.101, LOSO 15.73
  EDA知見: sp15 RMSE=39.2 が主因。高MC域(>150%)で過小予測が深刻。
  自動提出閾値: LOSO < 15.4 のみ提出（1日5回制限）

Run:
  python src/nir/nir_agent_v2.py              # auto-loop (max 5 iterations)
  python src/nir/nir_agent_v2.py --once       # 1 iteration
  python src/nir/nir_agent_v2.py --iters 3   # N iterations

Requires: GEMINI_API_KEY environment variable

Rate limits (Gemini free tier):
  gemini-2.5-pro:   5 RPM, 25 RPD
  gemini-2.0-flash: 15 RPM, 1500 RPD
  → API呼び出し間に MIN_CALL_INTERVAL 秒スリープ + 429時に指数バックオフ
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

# 実験レター管理 (M1/M2 は手動実験済み)
EXP_LETTERS = [f"Q{i}" for i in range(1, 20)] + [f"R{i}" for i in range(1, 20)]

MAX_RETRIES = 2
MAX_ITERS   = 5
LOSO_SUBMIT_THRESHOLD = 15.2   # これを下回ったら自動提出 (LBベスト15.395基準)

# ── Gemini クライアント & レート制限 ─────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# モデル選択 (Gemini free tier)
# gemini-2.0-flash: 15 RPM / 1500 RPD (最大枠)
PLANNER_MODEL  = "gemini-2.0-flash"
CODER_MODEL    = "gemini-2.0-flash"
ANALYZER_MODEL = "gemini-2.0-flash"

MIN_CALL_INTERVAL = 5    # 秒 (15RPM → 4s + 余裕)
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
def read_scores() -> str:
    return SCORES_MD.read_text(encoding="utf-8")


def update_scores_md(letter: str, hypothesis: str, loso: float | None,
                     submitted: bool, analysis: str) -> None:
    """Append agent experiment row to scores.md 打ち手ログ table."""
    content = SCORES_MD.read_text(encoding="utf-8")
    lines = content.split("\n")

    loso_str      = f"**{loso:.4f}**" if loso else "—"
    lb_str        = "提出済" if submitted else "—"
    delta_str     = f"{loso - 15.73:+.2f}" if loso else "—"
    hyp_short     = hypothesis[:55].replace("|", "/")
    analysis_short = analysis[:80].replace("|", "/").replace("\n", " ")
    new_row = (f"| **{letter}** | {hyp_short} | LGBM(I2-params) | — "
               f"| {loso_str} | {lb_str} | Agent delta={delta_str}. {analysis_short} |")

    # Insert after last table row
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

    for attempt in range(3):
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
                backoff = 30 * (2 ** attempt)
                print(f"[API] 429 Rate limit hit ({model}). Backing off {backoff}s ...")
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
    return "N99"


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

# ── I2ベースパイプライン (MSC+SG(w=9,p=2)+EPO(n=5)) ────────────────────────
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

# I2固定パラメータ
I2_PARAMS = {{**LGBM_BASE_PARAMS,
              "learning_rate": 0.02, "num_leaves": 63,
              "feature_fraction": 0.07, "min_child_samples": 10}}

data = load_data()
y_train = data["y_train"]; X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]; test_ids = data["test_ids"]; sp_train = data["sp_train"]

ref   = X_train_raw.mean(axis=0)
Xtr_sg = sg_deriv(msc(X_train_raw, ref), window=9, polyorder=2)
Xte_sg = sg_deriv(msc(X_test_raw,  ref), window=9, polyorder=2)
V      = compute_epo_matrix(Xtr_sg, y_train, sp_train, n_components=5)
Xtr    = apply_epo(Xtr_sg, V)
Xte    = apply_epo(Xte_sg, V)

# ===== EXPERIMENT CODE BELOW =====
# 注意事項:
#   - ターゲット変換は y_sqrt = np.sqrt(y_train) が I2 デフォルト
#   - LOSO評価は loso_folds(sp_train) を使う
#   - 提出前評価: print(f"LOSO-RMSE = {{rmse:.4f}}")  ← この形式必須
#   - save_submission + submit_to_signate(OUT, memo, loso=rmse) で提出
#     (LOSO >= {threshold} なら自動的にスキップされる)
#   - I2_PARAMS と Xtr, Xte, y_train, sp_train はすでに定義済み
#   - 新しいモデルや前処理の試みはこのセクションに書く
# ===== END INSTRUCTIONS =====
'''


# ── Node: planner ─────────────────────────────────────────────────────────────
def planner(state: DSState) -> DSState:
    print(f"\n{'='*60}")
    print(f"[PLANNER] Iteration {state['iteration']+1}/{MAX_ITERS}")
    print(f"{'='*60}")

    scores_text = read_scores()
    history_text = ""
    if state["history"]:
        history_text = "\n\nAgent session history:\n"
        for h in state["history"]:
            history_text += (
                f"  Exp {h['letter']}: {h['hypothesis'][:80]}\n"
                f"    LOSO={h.get('loso_score','?')}  delta={h.get('loso_delta','?')}\n"
                f"    Result: {h.get('analysis','')[:150]}\n\n"
            )

    system = (
        "You are an expert NIR spectroscopy data scientist competing on SIGNATE. "
        "Goal: minimize LOSO-RMSE (Leave-One-Species-Out CV) for wood moisture prediction from NIR spectra.\n\n"

        "=== CURRENT STATE ===\n"
        "LB best: P1 = 15.395 (LOSO=15.4725, gap=-0.077)\n"
        "Pipeline: MSC -> SG(w=9,p=2,deriv=1) -> EPO(n=5) -> y^0.27 -> LGBM(lr=0.02, leaves=63, ff=0.07, mcs=10)\n"
        "Auto-submit threshold: LOSO < 15.2 only\n\n"

        "=== EDA FINDINGS ===\n"
        "1. sp15: RMSE=39.20, bias=-6.80, n=112, MC_max=298.6% → dominant bottleneck\n"
        "2. sp11: RMSE=19.96, bias=+11.18 (opposite direction to sp15)\n"
        "3. MC帯別RMSE: 0-30%:low, 30-60%:mid, >150%:catastrophic underprediction\n"
        "4. EPO後 test-train距離≈0 → スペクトル空間の汎化は解決済み\n\n"

        "=== CONFIRMED OPTIMAL (do NOT change) ===\n"
        "- MSC scatter correction\n"
        "- SG window=9, poly=2\n"
        "- EPO n=5 (global, not fold-internal)\n"
        "- feature_fraction=0.07\n"
        "- lr=0.02, leaves=63, mcs=10\n"
        "- Target transform: y^0.27 (P1/P6で確定。0.265/0.278は悪化)\n"
        "- Loss: L2 (Huber/MAE/Tweedie全て悪化)\n\n"

        "=== FAILED APPROACHES (do NOT repeat) ===\n"
        "- NN系 (MLP, Transformer, CNN): LOSO良→LB悪の逆相関確定\n"
        "- PLS after EPO: LB=26.37\n"
        "- leaves=31,mcs=30: LOSO改善→LB悪化\n"
        "- Seed ensemble: r=0.998で多様性なし\n"
        "- XGBoost: leaf-wiseに劣る\n"
        "- sample_weight(高MC): p=0.27と相性不良(P5)\n"
        "- DART booster: 800 roundsで収束不足(P3)\n"
        "- Tweedie loss: L2より不利(P4)\n"
        "- p != 0.27: P6で0.270が真の最適と確定\n\n"

        "=== PROMISING DIRECTIONS (未試行) ===\n"
        "- sp15特化: 残差に対する2段階モデル(global + sp15補正)\n"
        "- MC域別スタッキング: 低MC/高MCで別モデル(B1失敗だが手法改良余地あり)\n"
        "- 特徴量エンジニアリング: EPO後スペクトルの積/比特徴量\n"
        "- 異なるbin_widthでのEPO再探索\n\n"

        "Respond ONLY with JSON: {\"hypothesis\": \"...\", \"title\": \"...\", \"exp_type\": \"...\"}"
    )

    user_msg = (
        f"Score tracker:\n{scores_text}"
        f"{history_text}\n\n"
        "Propose the single most promising next experiment. "
        "Focus on sp15 / high-MC extrapolation problem. "
        "Be specific about implementation. "
        "Respond ONLY with JSON."
    )

    raw = gemini(system, user_msg, model=PLANNER_MODEL, max_tokens=1024)
    # Strip markdown fences before JSON parsing
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
        # Fallback: extract "hypothesis" value with regex
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
        "AVAILABLE FUNCTIONS (exact signatures):\n"
        "  msc(X, reference=None) -> ndarray\n"
        "  snv(X) -> ndarray\n"
        "  sg_deriv(X, window=11, polyorder=2, deriv=1) -> ndarray\n"
        "  loso_folds(sp_train) -> yields (tr_idx, va_idx, sp)\n"
        "  loso_rmse(oof, y) -> float\n"
        "  loso_lgbm(X_tr, y, sp_train, params, n_rounds=1000) -> (rmse, avg_rounds, oof)\n"
        "  save_submission(test_ids, preds, path) -> None\n"
        "  submit_to_signate(path, memo, loso=float) -> None\n"
        "  LGBM_BASE_PARAMS, I2_PARAMS (already defined)\n"
        "  Xtr, Xte, y_train, sp_train, test_ids (already defined)\n\n"
        "RULES:\n"
        "1. DO NOT redefine: msc, snv, sg_deriv, loso_folds, loso_rmse, compute_epo_matrix, apply_epo, Xtr, Xte\n"
        "2. MUST print: print(f'LOSO-RMSE = {rmse:.4f}')\n"
        "3. MUST call: save_submission(test_ids, preds, OUT)\n"
        "4. MUST call: submit_to_signate(OUT, f'ExpXX: ... LOSO={rmse:.4f}', loso=rmse)\n"
        "5. Use lgb.train directly for custom objectives (not loso_lgbm wrapper)\n"
        "6. Return ONLY Python code, no markdown fences.\n\n"
        f"Template:\n{template}"
    )

    user_msg = (
        f"Implement: {hypo}{error_ctx}\n\n"
        "Fill in the EXPERIMENT CODE section only. "
        "Xtr/Xte/y_train/sp_train are already preprocessed with I2 pipeline. "
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
        "(1) LOSO vs I2 baseline (15.73)? "
        "(2) What does this tell us about sp15/high-MC problem? "
        "(3) What to try next?"
    )
    user_msg = (
        f"Exp {letter}: {state['hypothesis']}\n"
        f"LOSO-RMSE: {state['loso_score']} (I2 baseline=15.73, LB best=16.101)\n\n"
        f"Output:\n{state['logs'][-2000:]}"
    )

    analysis = gemini(system, user_msg, model=ANALYZER_MODEL, max_tokens=512)
    print(f"[ANALYZER] {analysis[:300]}")

    baseline = 15.73
    loso = state["loso_score"]
    delta = (loso - baseline) if loso else None
    submitted = loso is not None and loso < LOSO_SUBMIT_THRESHOLD
    history_entry = {
        "letter": letter,
        "hypothesis": state["hypothesis"],
        "loso_score": loso,
        "loso_delta": f"{delta:+.4f}" if delta is not None else "N/A",
        "analysis": analysis,
    }

    update_scores_md(letter, state["hypothesis"], loso, submitted, analysis)

    return {
        **state,
        "analysis": analysis,
        "history": state["history"] + [history_entry],
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

    # 手動実験の結果を履歴に反映
    initial_history = [
        {
            "letter": "M2",
            "hypothesis": "べき乗変換p=0.30 (LB=15.545)",
            "loso_score": 15.5877,
            "loso_delta": "-0.1423",
            "analysis": "p=0.30でLOSO=15.5877。LB=15.545。高MC域の圧縮が有効。",
        },
        {
            "letter": "P1",
            "hypothesis": "細粒度p探索でp=0.27が最良 (LB=15.395, 新ベスト)",
            "loso_score": 15.4725,
            "loso_delta": "-0.1152",
            "analysis": "p=0.27でLOSO=15.4725, LB=15.395(新ベスト)。gap=-0.077(LBがLOSOより良い)。"
                        "P6超細粒度でp=0.270が真の最適と確定。p=0.265/0.278は悪化。",
        },
        {
            "letter": "P2",
            "hypothesis": "sample_weight(MC>100%, w=2.0) + p=0.30",
            "loso_score": 15.5347,
            "loso_delta": "-0.0530",
            "analysis": "M2比-0.053の小改善のみ。P1(p=0.27)には遠く及ばず。",
        },
        {
            "letter": "P3",
            "hypothesis": "DART booster + p=0.30",
            "loso_score": 23.07,
            "loso_delta": "+7.49",
            "analysis": "固定rounds=800で収束不足。大幅悪化。DARTは不採用。",
        },
        {
            "letter": "P4",
            "hypothesis": "Tweedie目的関数(vp=1.2〜1.8)",
            "loso_score": 15.84,
            "loso_delta": "+0.27",
            "analysis": "全設定でM2より悪化。Tweedie不採用。",
        },
        {
            "letter": "P5",
            "hypothesis": "p=0.27 + sample_weight(MC>100%, w=2.0) 組み合わせ",
            "loso_score": 15.5368,
            "loso_delta": "+0.0643",
            "analysis": "P1(p=0.27)より悪化。重み付けとp=0.27の組み合わせは相乗効果なし。",
        },
        {
            "letter": "P6",
            "hypothesis": "p=0.270周辺超細粒度探索(0.255〜0.290, step=0.005)",
            "loso_score": 15.4725,
            "loso_delta": "+0.0000",
            "analysis": "p=0.270が真の最適と確定。これ以上の変換チューニングは限界。",
        },
    ]

    initial_state: DSState = {
        "iteration":   0,
        "exp_letter":  "Q1",
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
    print(f"NIR DS Agent v2  |  max_iters={max_iters}")
    print(f"LB best: P1=15.395 (LOSO=15.4725, p=0.27)")
    print(f"Submit threshold: LOSO < {LOSO_SUBMIT_THRESHOLD}")
    print("=" * 60)

    final = agent.invoke(initial_state, config={"recursion_limit": 300})

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    for h in final.get("history", []):
        submitted = "✓ SUBMITTED" if h.get("loso_score") and h["loso_score"] < LOSO_SUBMIT_THRESHOLD else ""
        print(f"  Exp {h['letter']}: LOSO={h.get('loso_score','?')} ({h.get('loso_delta','?')}) {submitted}")
        print(f"    {h['hypothesis'][:80]}")
    print("=" * 60)
