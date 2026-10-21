"""Train the v2 models on all training-month trips and save them for the Streamlit demo.

Uses exactly the configurations selected in reports/cv_v2_*.csv (the same ones final_eval_v2.py evaluated on June),
plus the frozen CQR shift. Writes models/app_models.joblib (not committed; ~1 hour on 12 threads because of the
quantile models). ``--quick`` trains small models on a 200k sample in about a minute, for development only.

Usage: python scripts/train_app_models.py [--quick]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from taxi import config, data, features, models, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

LO, HI = 0.1, 0.9
MODELS_DIR = config.ROOT / "models"


def parse(label):
    m = re.search(r"leaves=(\d+)\) @(\d+) trees", label)
    return int(m.group(1)), int(m.group(2))


point_cv = pd.read_csv(config.REPORTS_DIR / "cv_v2_point.csv")
point_label = point_cv[~point_cv["model"].str.startswith("baseline")].sort_values("mae_min_mean").iloc[0]["model"]
int_row = pd.read_csv(config.REPORTS_DIR / "cv_v2_intervals.csv").sort_values("width_cqr_min_mean").iloc[0]
leaves, n = parse(point_label)
leaves_q, n_q = parse(int_row["model"])
if args.quick:
    n, n_q = 200, 200

zones = zmod.load_zones()
train, _ = data.load_clean(config.TRAIN_MONTH, max_rows=200_000 if args.quick else None)
X, y = features.build_features(train, zones), train["duration_s"].to_numpy("float64")
w = models.recency_weights(train["pickup"], 7)

fitted = {}
for key, m in [
    ("point", models.LGBMModel("raw", "regression", n_estimators=n, num_leaves=leaves, recency_halflife_days=7)),
    ("lo", models.LGBMModel("raw", "quantile", n_estimators=n_q, num_leaves=leaves_q, recency_halflife_days=7, alpha=LO)),
    ("hi", models.LGBMModel("raw", "quantile", n_estimators=n_q, num_leaves=leaves_q, recency_halflife_days=7, alpha=HI)),
]:
    t0 = time.time()
    fitted[key] = m.fit(X, y, sample_weight=w)
    print(f"{key}: {time.time() - t0:.0f}s", flush=True)

june = json.loads((config.REPORTS_DIR / "final_results_v2.json").read_text())
MODELS_DIR.mkdir(exist_ok=True)
out = MODELS_DIR / ("app_models_quick.joblib" if args.quick else "app_models.joblib")
joblib.dump({
    **fitted,
    "cqr_shift_s": float(int_row["cqr_q_all_folds_s"]),
    "zones": zones,
    "meta": {
        "quick": args.quick, "train_month": config.TRAIN_MONTH, "n_train": len(train),
        "point_model": point_label, "interval_model": int_row["model"],
        "june_mae_min": next(r["mae_min"] for r in june["results"] if r["model"].startswith("v2")),
        "june_coverage": june["intervals"]["coverage_cqr"],
    },
}, out, compress=3)
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
