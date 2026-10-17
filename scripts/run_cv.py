"""Time-based CV on the training month: baselines, Ridge, LightGBM.

Writes reports/cv_results.csv, reports/cv_results.md and out-of-fold predictions to data/oof/.
The held-out test month is never read here.

Usage: python scripts/run_cv.py [--quick] [--only SUBSTRING]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from taxi import config, cv, data, features, models, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="tiny subsample for a fast smoke run")
parser.add_argument("--only", default="", help="run only experiments whose name contains this text")
parser.add_argument("--out", default="cv_results", help="output basename in reports/")
args = parser.parse_args()

n_rows = 30_000 if args.quick else config.MAX_TRAIN_ROWS
config.REPORTS_DIR.mkdir(exist_ok=True)
oof_dir = config.DATA_DIR / "oof"
oof_dir.mkdir(parents=True, exist_ok=True)

zones = zmod.load_zones()
full, _ = data.load_clean(config.TRAIN_MONTH)
sample = data.subsample(full, n_rows)
month_start, _ = data.month_bounds(config.TRAIN_MONTH)
print(f"train month cleaned rows {len(full):,}; CV sample {len(sample):,}")

X = features.build_features(sample, zones, geo=True)
y = sample["duration_s"].to_numpy(dtype="float64")
folds = cv.time_folds(sample["pickup"], month_start)
for f in folds:
    print(f"fold {f.k}: train {len(f.train_idx):,} val {len(f.val_idx):,} ({f.val_start.date()} - {f.val_end.date()})")

NO_GEO = [c for c in features.GEO_FEATURES]

# name -> (factory, X-variant, iteration checkpoints for LightGBM or None)
EXPERIMENTS = [
    ("baseline: global median", lambda: models.GlobalMedian(), "geo", None),
    ("baseline: route-hour median", lambda: models.RouteHourMedian(5), "geo", None),
    ("ridge a=1 (no geo)", lambda: models.RidgeModel(1.0, geo=False), "nogeo", None),
    ("ridge a=10 (no geo)", lambda: models.RidgeModel(10.0, geo=False), "nogeo", None),
    ("ridge a=10 (+geo)", lambda: models.RidgeModel(10.0, geo=True), "geo", None),
    ("ridge a=100 (+geo)", lambda: models.RidgeModel(100.0, geo=True), "geo", None),
    ("ridge a=10 (+geo, smearing)", lambda: models.RidgeModel(10.0, geo=True, smearing=True), "geo", None),
    ("lgbm log-L2 (no geo)", lambda: models.LGBMModel("log", "regression", n_estimators=1000), "nogeo", [300, 600, 1000]),
    ("lgbm log-L2 (+geo)", lambda: models.LGBMModel("log", "regression", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L1 (+geo)", lambda: models.LGBMModel("raw", "regression_l1", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L2 (+geo)", lambda: models.LGBMModel("raw", "regression", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L1 (+geo, leaves=31)", lambda: models.LGBMModel("raw", "regression_l1", n_estimators=1000, num_leaves=31), "geo", [300, 600, 1000]),
    ("lgbm raw-L1 (+geo, leaves=255, mcs=100)", lambda: models.LGBMModel("raw", "regression_l1", n_estimators=1000, num_leaves=255, min_child_samples=100), "geo", [300, 600, 1000]),
]

rows = []
for name, factory, variant, iters in EXPERIMENTS:
    if args.only and args.only not in name:
        continue
    Xv = X.drop(columns=NO_GEO) if variant == "nogeo" else X
    checkpoints = iters or [None]
    per_fold = {c: [] for c in checkpoints}
    oof = {c: [] for c in checkpoints}
    t0 = time.time()
    for f in folds:
        m = factory().fit(Xv.iloc[f.train_idx], y[f.train_idx])
        for c in checkpoints:
            p = m.predict(Xv.iloc[f.val_idx], num_iteration=c) if c else m.predict(Xv.iloc[f.val_idx])
            per_fold[c].append(cv.metrics(y[f.val_idx], p))
            oof[c].append(pd.DataFrame({"idx": f.val_idx, "fold": f.k, "y": y[f.val_idx], "pred": p}))
    for c in checkpoints:
        label = name if c is None else f"{name} @{c} trees"
        s = cv.summarize(per_fold[c])
        rows.append({"model": label, **s, "per_fold_mae": [round(d["mae_min"], 3) for d in per_fold[c]]})
        slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
        pd.concat(oof[c]).to_parquet(oof_dir / f"{slug}.parquet")
        print(f"{label:60s} MAE {s['mae_min_mean']:.3f}±{s['mae_min_std']:.3f}  RMSE {s['rmse_min_mean']:.3f}±{s['rmse_min_std']:.3f}")
    print(f"   ({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
res.to_csv(config.REPORTS_DIR / f"{args.out}.csv", index=False)
lines = ["| model | MAE min (mean ± std) | RMSE min (mean ± std) | per-fold MAE |", "|---|---|---|---|"]
for r in res.itertuples():
    lines.append(f"| {r.model} | {r.mae_min_mean:.3f} ± {r.mae_min_std:.3f} | {r.rmse_min_mean:.3f} ± {r.rmse_min_std:.3f} | {r.per_fold_mae} |")
(config.REPORTS_DIR / f"{args.out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
