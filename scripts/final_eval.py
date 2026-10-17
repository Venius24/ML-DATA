"""Final evaluation on the frozen held-out month. Run ONCE.

The model is chosen programmatically from reports/cv_results.csv (lowest mean CV MAE among non-baseline
rows); nothing here is tuned on the test month. The script refuses to run again once
reports/final_results.json exists. ``--dry-run`` exercises the same code on the last CV block of the
training month and never opens the test file.

Usage: python scripts/final_eval.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from taxi import config, cv, data, experiments, features, models, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

NL = chr(10)
out_json = config.REPORTS_DIR / ("final_dryrun.json" if args.dry_run else "final_results.json")
out_md = out_json.with_suffix(".md")
if not args.dry_run and out_json.exists():
    sys.exit(f"{out_json.name} already exists: the test month has been evaluated once and is frozen.")

# ---- choose the final model from CV only -------------------------------------------------------
cvres = pd.read_csv(config.REPORTS_DIR / "cv_results.csv")
cand = cvres[~cvres["model"].str.startswith("baseline")]
final_label = cand.sort_values("mae_min_mean").iloc[0]["model"]
ridge_label = cand[cand["model"].str.startswith("ridge")].sort_values("mae_min_mean").iloc[0]["model"]
print("final model (lowest CV MAE):", final_label, "| best ridge:", ridge_label)

zones = zmod.load_zones()
train_full, _ = data.load_clean(config.TRAIN_MONTH)
sample = data.subsample(train_full, config.MAX_TRAIN_ROWS)

if args.dry_run:
    month_start, _ = data.month_bounds(config.TRAIN_MONTH)
    f = cv.time_folds(sample["pickup"], month_start)[-1]
    train_df, test_df, test_steps = sample.iloc[f.train_idx], sample.iloc[f.val_idx], []
    train_full = train_full[train_full["pickup"] < f.val_start - pd.Timedelta(seconds=data.MAX_DURATION_S)]
    test_name = f"DRY RUN: last CV block of {config.TRAIN_MONTH}"
else:
    train_df = sample
    test_df, test_steps = data.load_clean(config.TEST_MONTH)  # first and only read of the test month
    test_name = config.TEST_MONTH

Xtr = features.build_features(train_df, zones)
Xte = features.build_features(test_df, zones)
ytr = train_df["duration_s"].to_numpy("float64")
yte = test_df["duration_s"].to_numpy("float64")
Xfull = features.build_features(train_full, zones)
yfull = train_full["duration_s"].to_numpy("float64")


def fit_label(label, X, y, pickup):
    factory, variant, n_trees = experiments.resolve(label)
    m = factory()
    if n_trees:
        m.params["n_estimators"] = n_trees
    cols = X.drop(columns=experiments.NO_GEO) if variant == "nogeo" else X
    hl = getattr(m, "recency_halflife_days", None)
    kw = {"sample_weight": models.recency_weights(pickup, hl)} if hl else {}
    return m.fit(cols, y, **kw), variant


def predict(m, variant, X):
    return m.predict(X.drop(columns=experiments.NO_GEO) if variant == "nogeo" else X)


preds = {}
preds["global median"] = models.GlobalMedian().fit(Xtr, ytr).predict(Xte)
preds["route-hour median (same sample)"] = models.RouteHourMedian(5).fit(Xtr, ytr).predict(Xte)
preds["route-hour median (all training-month trips)"] = models.RouteHourMedian(5).fit(Xfull, yfull).predict(Xte)
m, v = fit_label(ridge_label, Xtr, ytr, train_df["pickup"])
preds[f"best Ridge: {ridge_label}"] = predict(m, v, Xte)
m, v = fit_label(final_label, Xtr, ytr, train_df["pickup"])
preds[f"FINAL: {final_label}"] = predict(m, v, Xte)
final_key = f"FINAL: {final_label}"

rows = []
for k, p in preds.items():
    rows.append({"model": k, **cv.metrics(yte, p), "bias_min": float((p - yte).mean() / 60)})
res = pd.DataFrame(rows)

day = test_df["pickup"].dt.date
daily = pd.DataFrame({"day": day, "dow": test_df["pickup"].dt.day_name(), "n": 1,
                      "final_ae": np.abs(preds[final_key] - yte) / 60,
                      "base_ae": np.abs(preds["route-hour median (same sample)"] - yte) / 60,
                      "y": yte / 60}).groupby("day").agg(
    dow=("dow", "first"), trips=("n", "sum"), final_mae=("final_ae", "mean"), baseline_mae=("base_ae", "mean"),
    mean_duration=("y", "mean")).reset_index()
hol = sorted(features.us_holiday_dates(sorted(test_df["pickup"].dt.year.unique().tolist())))
daily["holiday"] = daily["day"].isin(hol)

lines = [
    f"# Final evaluation on {test_name}",
    "",
    f"Training: {len(train_df):,} trips sampled from {config.TRAIN_MONTH} (seed {config.SEED}); "
    f"evaluation: {len(test_df):,} cleaned trips. Final model chosen by lowest mean CV MAE: `{final_label}`.",
    "",
    "| model | MAE (min) | RMSE (min) | mean signed error (min) |", "|---|---:|---:|---:|",
]
for r in res.itertuples():
    lines.append(f"| {r.model} | {r.mae_min:.3f} | {r.rmse_min:.3f} | {r.bias_min:+.2f} |")
lines += ["", "## Test-month cleaning log", "", "| step | removed | remaining |", "|---|---:|---:|"]
lines += [f"| {s.name} | {s.removed:,} | {s.remaining:,} |" for s in test_steps]
lines += ["", "## Per-day MAE (minutes)", "", "| day | weekday | trips | final | baseline (route-hour) | mean duration | US holiday |", "|---|---|---:|---:|---:|---:|---|"]
for r in daily.itertuples():
    lines.append(f"| {r.day} | {r.dow} | {r.trips:,} | {r.final_mae:.2f} | {r.baseline_mae:.2f} | {r.mean_duration:.2f} | {'yes' if r.holiday else ''} |")
out_md.write_text(NL.join(lines) + NL, encoding="utf-8")
out_json.write_text(json.dumps({
    "test": test_name, "final_model": final_label, "n_train": len(train_df), "n_test": len(test_df),
    "results": res.to_dict(orient="records"),
    "daily": daily.assign(day=daily["day"].astype(str)).to_dict(orient="records")}, indent=1))
print(NL.join(lines))
