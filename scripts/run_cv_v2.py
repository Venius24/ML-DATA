"""v2 CV: every cleaned trip of the training month (no 500k cap) plus 80% prediction intervals.

Same time-based folds as run_cv.py. Point models: the v1 winner (raw-L2, geometry, recency 7d) with more
trees and bigger leaves, since more data can support a larger model. Intervals: LightGBM quantile models for
the 10th and 90th percentile, evaluated raw and after CQR calibration (the correction for fold k is learned
on the other folds' out-of-fold predictions, so the reported coverage is honest).
The held-out months are never read here.

Writes reports/cv_v2.md, reports/cv_v2_point.csv, reports/cv_v2_intervals.csv; OOF files go to data/oof_v2/.

Usage: python scripts/run_cv_v2.py [--quick] [--only SUBSTRING] [--skip-existing]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from taxi import config, cv, data, features, intervals, models, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="30k-row subsample and few trees, for a smoke run")
parser.add_argument("--only", default="", help="run only experiments whose name contains this text")
parser.add_argument("--skip-existing", action="store_true", help="do not refit experiments whose OOF files exist")
args = parser.parse_args()

NL = chr(10)
LO, HI = 0.1, 0.9
TARGET = HI - LO
scale = 0.1 if args.quick else 1.0


def trees(*ns):
    return [max(int(n * scale), 10) for n in ns]


POINT = [
    ("lgbm raw-L2 (+geo, recency 7d, leaves=127)", dict(num_leaves=127), trees(600, 1000, 1500)),
    ("lgbm raw-L2 (+geo, recency 7d, leaves=255)", dict(num_leaves=255), trees(600, 1000, 1500)),
]
QUANTILE = ("lgbm quantile (+geo, recency 7d, leaves=127)", dict(num_leaves=127), trees(600, 1000))


def point_model(kw, n):
    return models.LGBMModel("raw", "regression", n_estimators=n, recency_halflife_days=7, **kw)


def quantile_model(kw, n, alpha):
    return models.LGBMModel("raw", "quantile", n_estimators=n, recency_halflife_days=7, alpha=alpha, **kw)


def slug(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


zones = zmod.load_zones()
full, _ = data.load_clean(config.TRAIN_MONTH)
sample = data.subsample(full, 30_000) if args.quick else full
month_start, _ = data.month_bounds(config.TRAIN_MONTH)
X = features.build_features(sample, zones)
y = sample["duration_s"].to_numpy("float64")
folds = cv.time_folds(sample["pickup"], month_start)
print(f"CV on {len(sample):,} trips; fold train sizes {[len(f.train_idx) for f in folds]}", flush=True)

oof_dir = config.DATA_DIR / ("oof_v2_quick" if args.quick else "oof_v2")
oof_dir.mkdir(parents=True, exist_ok=True)
out_dir = oof_dir if args.quick else config.REPORTS_DIR


def run(name, make, checkpoints):
    """Fit one model per fold, save OOF predictions at each tree checkpoint, return {label: path}."""
    labels = {c: f"{name} @{c} trees" for c in checkpoints}
    paths = {c: oof_dir / f"{slug(labels[c])}.parquet" for c in checkpoints}
    if args.skip_existing and all(p.exists() for p in paths.values()):
        return {labels[c]: paths[c] for c in checkpoints}
    t0 = time.time()
    parts = {c: [] for c in checkpoints}
    for f in folds:
        m = make()
        w = models.recency_weights(sample["pickup"].iloc[f.train_idx], m.recency_halflife_days)
        m.fit(X.iloc[f.train_idx], y[f.train_idx], sample_weight=w)
        for c in checkpoints:
            p = m.predict(X.iloc[f.val_idx], num_iteration=c)
            parts[c].append(pd.DataFrame({"idx": f.val_idx, "fold": f.k, "y": y[f.val_idx], "pred": p}))
    for c in checkpoints:
        pd.concat(parts[c]).to_parquet(paths[c])
    print(f"{name}: fitted in {time.time() - t0:.0f}s", flush=True)
    return {labels[c]: paths[c] for c in checkpoints}


# ---- point models --------------------------------------------------------------------------------
point_rows = []
base_parts = []
for f in folds:
    b = models.RouteHourMedian(5).fit(X.iloc[f.train_idx], y[f.train_idx])
    base_parts.append(cv.metrics(y[f.val_idx], b.predict(X.iloc[f.val_idx])))
point_rows.append({"model": "baseline: route-hour median (all trips)", **cv.summarize(base_parts),
                   "per_fold_mae": [round(d["mae_min"], 3) for d in base_parts]})
for name, kw, cps in POINT:
    if args.only and args.only not in name:
        continue
    for label, path in run(name, lambda: point_model(kw, max(cps)), cps).items():
        o = pd.read_parquet(path)
        per_fold = [cv.metrics(g["y"], g["pred"]) for _, g in o.groupby("fold")]
        point_rows.append({"model": label, **cv.summarize(per_fold), "per_fold_mae": [round(d["mae_min"], 3) for d in per_fold]})
        print(f"{label:60s} MAE {point_rows[-1]['mae_min_mean']:.3f}", flush=True)
point = pd.DataFrame(point_rows)
point.to_csv(out_dir / "cv_v2_point.csv", index=False)

# ---- intervals -----------------------------------------------------------------------------------
int_rows = []
name, kw, cps = QUANTILE
if not args.only or args.only in name:
    lo_paths = run(f"{name} q{LO}", lambda: quantile_model(kw, max(cps), LO), cps)
    hi_paths = run(f"{name} q{HI}", lambda: quantile_model(kw, max(cps), HI), cps)
    for c, (lp, hp) in zip(cps, zip(lo_paths.values(), hi_paths.values())):
        lo, hi = pd.read_parquet(lp), pd.read_parquet(hp)
        d = lo.rename(columns={"pred": "lo"}).assign(hi=hi["pred"].to_numpy())
        per = []
        for k, g in d.groupby("fold"):
            rest = d[d["fold"] != k]
            q = intervals.cqr_correction(rest["y"], rest["lo"], rest["hi"], TARGET)
            clo, chi = intervals.apply_correction(g["lo"], g["hi"], q, floor=data.MIN_DURATION_S)
            per.append({
                "fold": k,
                "coverage_raw": intervals.coverage(g["y"], g["lo"], g["hi"]),
                "width_raw_min": intervals.mean_width(g["lo"], g["hi"]) / 60,
                "pinball_lo_min": intervals.pinball(g["y"], g["lo"], LO) / 60,
                "pinball_hi_min": intervals.pinball(g["y"], g["hi"], HI) / 60,
                "cqr_q_min": q / 60,
                "coverage_cqr": intervals.coverage(g["y"], clo, chi),
                "width_cqr_min": intervals.mean_width(clo, chi) / 60,
            })
        per = pd.DataFrame(per)
        q_all = intervals.cqr_correction(d["y"], d["lo"], d["hi"], TARGET)
        row = {"model": f"{name} @{c} trees", "cqr_q_all_folds_s": q_all}
        for col in per.columns.drop("fold"):
            row[f"{col}_mean"], row[f"{col}_std"] = per[col].mean(), per[col].std(ddof=1)
        row["per_fold_coverage_cqr"] = [round(v, 3) for v in per["coverage_cqr"]]
        int_rows.append(row)
        print(f"{row['model']:60s} coverage raw {row['coverage_raw_mean']:.3f} cqr {row['coverage_cqr_mean']:.3f} "
              f"width cqr {row['width_cqr_min_mean']:.2f} min", flush=True)
ints = pd.DataFrame(int_rows)
if len(ints):
    ints.to_csv(out_dir / "cv_v2_intervals.csv", index=False)

# ---- report --------------------------------------------------------------------------------------
lines = [
    f"# CV v2: all {len(sample):,} cleaned trips of {config.TRAIN_MONTH} + {TARGET:.0%} prediction intervals", "",
    "Same 5 time-based folds as `cv_results.md`. Generated by `scripts/run_cv_v2.py`.", "",
    "## Point models", "",
    "| model | MAE min (mean ± std) | RMSE min (mean ± std) | per-fold MAE |", "|---|---|---|---|",
]
for r in point.itertuples():
    lines.append(f"| {r.model} | {r.mae_min_mean:.3f} ± {r.mae_min_std:.3f} | {r.rmse_min_mean:.3f} ± {r.rmse_min_std:.3f} | {r.per_fold_mae} |")
if len(ints):
    lines += [
        "", f"## {TARGET:.0%} intervals ({LO:.0%} and {HI:.0%} quantile models)", "",
        "Raw = the two quantile predictions as they are. CQR = both ends moved by one constant learned on the other folds.", "",
        "| model | coverage raw | width raw, min | coverage CQR | width CQR, min | CQR shift, min | per-fold coverage CQR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ints.itertuples():
        lines.append(f"| {r.model} | {r.coverage_raw_mean:.3f} ± {r.coverage_raw_std:.3f} | {r.width_raw_min_mean:.2f} | "
                     f"{r.coverage_cqr_mean:.3f} ± {r.coverage_cqr_std:.3f} | {r.width_cqr_min_mean:.2f} | "
                     f"{r.cqr_q_min_mean:+.2f} | {r.per_fold_coverage_cqr} |")
(out_dir / "cv_v2.md").write_text(NL.join(lines) + NL, encoding="utf-8")
print(NL.join(lines))
