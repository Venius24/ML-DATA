"""v2 final evaluation on a second frozen month (2026-06). Run ONCE.

Everything is chosen from reports/cv_v2_*.csv (training month only):
- point model: lowest mean CV MAE among non-baseline rows of cv_v2_point.csv;
- interval model: the quantile checkpoint with the narrowest mean CQR-calibrated width in cv_v2_intervals.csv,
  with the CQR shift learned on all out-of-fold predictions.
All models are trained on every cleaned trip of the training month. The v1 final model (500k sample) and the
baselines are evaluated on the same month for comparison. 2026-05 is not touched: it stays the v1 holdout.

Refuses to run again once reports/final_results_v2.json exists. ``--dry-run`` uses the last CV block of the
training month instead and never opens the test file.

Usage: python scripts/final_eval_v2.py [--dry-run]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from taxi import config, cv, data, features, intervals, models, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

NL = chr(10)
LO, HI = 0.1, 0.9
out_json = config.REPORTS_DIR / ("final_v2_dryrun.json" if args.dry_run else "final_results_v2.json")
out_md = out_json.with_suffix(".md")
if not args.dry_run and out_json.exists():
    sys.exit(f"{out_json.name} already exists: {config.TEST_MONTH_V2} has been evaluated once and is frozen.")


def parse(label):
    m = re.search(r"leaves=(\d+)\) @(\d+) trees", label)
    return int(m.group(1)), int(m.group(2))


# ---- choose everything from CV ------------------------------------------------------------------
point_cv = pd.read_csv(config.REPORTS_DIR / "cv_v2_point.csv")
point_label = point_cv[~point_cv["model"].str.startswith("baseline")].sort_values("mae_min_mean").iloc[0]["model"]
int_cv = pd.read_csv(config.REPORTS_DIR / "cv_v2_intervals.csv")
int_row = int_cv.sort_values("width_cqr_min_mean").iloc[0]
int_label, cqr_q = int_row["model"], float(int_row["cqr_q_all_folds_s"])
print("point model:", point_label, "| interval model:", int_label, f"| CQR shift {cqr_q / 60:+.2f} min")

zones = zmod.load_zones()
train, _ = data.load_clean(config.TRAIN_MONTH)
if args.dry_run:
    month_start, _ = data.month_bounds(config.TRAIN_MONTH)
    f = cv.time_folds(train["pickup"], month_start)[-1]
    train, test, test_steps = train.iloc[f.train_idx], train.iloc[f.val_idx], []
    test_name = f"DRY RUN: last CV block of {config.TRAIN_MONTH}"
else:
    test, test_steps = data.load_clean(config.TEST_MONTH_V2)  # first and only read of this month
    test_name = config.TEST_MONTH_V2
v1_train = data.subsample(train, config.MAX_TRAIN_ROWS)

Xtr, ytr = features.build_features(train, zones), train["duration_s"].to_numpy("float64")
Xv1, yv1 = features.build_features(v1_train, zones), v1_train["duration_s"].to_numpy("float64")
Xte, yte = features.build_features(test, zones), test["duration_s"].to_numpy("float64")
w = models.recency_weights(train["pickup"], 7)

preds = {
    "global median": models.GlobalMedian().fit(Xtr, ytr).predict(Xte),
    "route-hour median (all training-month trips)": models.RouteHourMedian(5).fit(Xtr, ytr).predict(Xte),
}
v1 = models.LGBMModel("raw", "regression", n_estimators=600, recency_halflife_days=7)
preds["v1 final model (500k sample, 600 trees)"] = v1.fit(
    Xv1, yv1, sample_weight=models.recency_weights(v1_train["pickup"], 7)).predict(Xte)
leaves, n = parse(point_label)
v2 = models.LGBMModel("raw", "regression", n_estimators=n, num_leaves=leaves, recency_halflife_days=7)
v2_key = f"v2 final model: {point_label}, all trips"
preds[v2_key] = v2.fit(Xtr, ytr, sample_weight=w).predict(Xte)

leaves_q, n_q = parse(int_label)
q_lo = models.LGBMModel("raw", "quantile", n_estimators=n_q, num_leaves=leaves_q, recency_halflife_days=7, alpha=LO)
q_hi = models.LGBMModel("raw", "quantile", n_estimators=n_q, num_leaves=leaves_q, recency_halflife_days=7, alpha=HI)
lo_raw = q_lo.fit(Xtr, ytr, sample_weight=w).predict(Xte)
hi_raw = q_hi.fit(Xtr, ytr, sample_weight=w).predict(Xte)
lo, hi = intervals.apply_correction(lo_raw, hi_raw, cqr_q, floor=data.MIN_DURATION_S)

res = pd.DataFrame([{"model": k, **cv.metrics(yte, p), "bias_min": float((p - yte).mean() / 60)} for k, p in preds.items()])
ints = {
    "coverage_raw": intervals.coverage(yte, lo_raw, hi_raw), "width_raw_min": intervals.mean_width(lo_raw, hi_raw) / 60,
    "coverage_cqr": intervals.coverage(yte, lo, hi), "width_cqr_min": intervals.mean_width(lo, hi) / 60,
    "below_interval": float(np.mean(yte < lo)), "above_interval": float(np.mean(yte > hi)),
}

dist_mi = Xte["dist_ft"].to_numpy() / 5280
bins = pd.cut(dist_mi, [0, 1, 2, 4, 8, np.inf], include_lowest=True, labels=["<=1", "1-2", "2-4", "4-8", ">8"])
by_dist = pd.DataFrame({"bin": bins, "inside": (yte >= lo) & (yte <= hi), "width": (hi - lo) / 60}).groupby(
    "bin", observed=True).agg(trips=("inside", "size"), coverage=("inside", "mean"), width_min=("width", "mean")).reset_index()

base_key = "route-hour median (all training-month trips)"
daily = pd.DataFrame({
    "day": test["pickup"].dt.date.to_numpy(), "dow": test["pickup"].dt.day_name().to_numpy(),
    "v2_ae": np.abs(preds[v2_key] - yte) / 60, "v1_ae": np.abs(preds["v1 final model (500k sample, 600 trees)"] - yte) / 60,
    "base_ae": np.abs(preds[base_key] - yte) / 60, "inside": (yte >= lo) & (yte <= hi),
}).groupby("day").agg(dow=("dow", "first"), trips=("inside", "size"), v2_mae=("v2_ae", "mean"), v1_mae=("v1_ae", "mean"),
                       baseline_mae=("base_ae", "mean"), coverage=("inside", "mean")).reset_index()
hol = features.us_holiday_dates(sorted(test["pickup"].dt.year.unique().tolist()))
daily["holiday"] = daily["day"].isin(hol)

lines = [
    f"# v2 final evaluation on {test_name}", "",
    f"Training: all {len(train):,} cleaned trips of {config.TRAIN_MONTH} (the v1 model: {len(v1_train):,}-trip sample). "
    f"Evaluation: {len(test):,} cleaned trips. Chosen from CV only (`reports/cv_v2.md`). Generated by `scripts/final_eval_v2.py`.", "",
    "## Point predictions", "",
    "| model | MAE (min) | RMSE (min) | mean signed error (min) |", "|---|---:|---:|---:|",
]
lines += [f"| {r.model} | {r.mae_min:.3f} | {r.rmse_min:.3f} | {r.bias_min:+.2f} |" for r in res.itertuples()]
lines += [
    "", f"## 80% prediction interval: `{int_label}`, CQR shift {cqr_q / 60:+.2f} min", "",
    "| | coverage | mean width (min) |", "|---|---:|---:|",
    f"| raw quantile models | {ints['coverage_raw']:.3f} | {ints['width_raw_min']:.2f} |",
    f"| after CQR calibration | {ints['coverage_cqr']:.3f} | {ints['width_cqr_min']:.2f} |", "",
    f"Trips below the interval: {ints['below_interval']:.1%}; above it: {ints['above_interval']:.1%} (target 10% each).", "",
    "### Coverage by zone-centroid distance (miles)", "", "| distance | trips | coverage | mean width (min) |", "|---|---:|---:|---:|",
]
lines += [f"| {r.bin} | {r.trips:,} | {r.coverage:.3f} | {r.width_min:.2f} |" for r in by_dist.itertuples()]
lines += ["", "## Test-month cleaning log", "", "| step | removed | remaining |", "|---|---:|---:|"]
lines += [f"| {s.name} | {s.removed:,} | {s.remaining:,} |" for s in test_steps]
lines += ["", "## Per day", "", "| day | weekday | trips | v2 MAE | v1 MAE | baseline MAE | interval coverage | US holiday |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
lines += [f"| {r.day} | {r.dow} | {r.trips:,} | {r.v2_mae:.2f} | {r.v1_mae:.2f} | {r.baseline_mae:.2f} | {r.coverage:.3f} | "
          f"{'yes' if r.holiday else ''} |" for r in daily.itertuples()]
out_md.write_text(NL.join(lines) + NL, encoding="utf-8")
out_json.write_text(json.dumps({
    "test": test_name, "point_model": point_label, "interval_model": int_label, "cqr_shift_s": cqr_q,
    "n_train": len(train), "n_test": len(test), "results": res.to_dict(orient="records"), "intervals": ints,
    "coverage_by_distance": by_dist.assign(bin=by_dist["bin"].astype(str)).to_dict(orient="records"),
    "daily": daily.assign(day=daily["day"].astype(str)).to_dict(orient="records")}, indent=1))
print(NL.join(lines))
