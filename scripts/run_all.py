"""Run the whole pipeline end to end (cross-platform alternative to `make all`).

Usage: python scripts/run_all.py [--skip-final]

final_eval.py refuses to run if reports/final_results.json already exists (the test month is frozen);
delete that file first if you really want to reproduce the held-out numbers from scratch.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/download_data.py"],
    ["scripts/eda.py"],
    ["scripts/run_cv.py"],
    # error analysis before the recency fix (the model chosen before the fix) and after it (CV winner)
    ["scripts/error_analysis.py", "--model", "lgbm raw-L2 (+geo) @600 trees"],
    ["scripts/error_analysis.py", "--out", "error_analysis_after_fix"],
    ["scripts/learning_curve.py"],
    ["scripts/final_eval.py"],
    ["scripts/summary.py"],
]

if __name__ == "__main__":
    skip = "--skip-final" in sys.argv
    steps = [st for st in STEPS if not (skip and st[0] in ("scripts/final_eval.py", "scripts/summary.py"))]
    for step in steps:
        print("==>", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], cwd=ROOT, check=True)
