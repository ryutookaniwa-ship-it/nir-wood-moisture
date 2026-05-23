"""
実験自動実行スクリプト
======================
1. Q1 (EPO bin_width探索)
2. Q2 (leaves/mcs再チューニング)
3. Q3 (EPO n_comp再探索)
4. 16:00 JST まで待機
5. agent --iters 5 を起動
"""
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(r"C:\Users\ryuch\OneDrive\デスクトップ\my_kaggle_project")
PYTHON = sys.executable
JST = timezone(timedelta(hours=9))

EXPERIMENTS = [
    BASE / "src" / "nir" / "nir_exp_Q1_epo_binwidth.py",
    BASE / "src" / "nir" / "nir_exp_Q2_leaves_mcs.py",
    BASE / "src" / "nir" / "nir_exp_Q3_epo_ncomp.py",
]

def log(msg):
    ts = datetime.now(JST).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── 手動実験を順番に実行 ─────────────────────────────────────────────────────
for script in EXPERIMENTS:
    log(f"Starting: {script.name}")
    result = subprocess.run([PYTHON, str(script)], cwd=str(BASE))
    if result.returncode == 0:
        log(f"Done: {script.name}")
    else:
        log(f"ERROR in {script.name} (exit={result.returncode}) - continuing")

# ── 16:00 JST まで待機 ───────────────────────────────────────────────────────
TARGET_HOUR = 16  # JST
now = datetime.now(JST)
target = now.replace(hour=TARGET_HOUR, minute=5, second=0, microsecond=0)
if now >= target:
    target = target  # already past 16:00, run immediately
else:
    wait_sec = (target - now).total_seconds()
    log(f"Experiments done. Waiting {wait_sec/60:.1f} min until {target.strftime('%H:%M')} JST for Gemini reset...")
    time.sleep(wait_sec)

# ── Agent 起動 ───────────────────────────────────────────────────────────────
log("Gemini rate limit should be reset. Starting agent --iters 5 ...")
result = subprocess.run(
    [PYTHON, str(BASE / "src" / "nir" / "nir_agent_v2.py"), "--iters", "5"],
    cwd=str(BASE),
)
log(f"Agent finished (exit={result.returncode})")
