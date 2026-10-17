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
parser.add_argument("--summarize-only", action="store_true", help="rebuild the table from existing OOF files, fit nothing")
parser.add_argument("--skip-existing", action="store_true", help="do not refit experiments whose OOF files exist")
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

from taxi.experiments import EXPERIMENTS, NO_GEO, labels_for  # noqa: E402


def slug(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


NL = chr(10)


def write_results(rows):
    res = pd.DataFrame(rows)
    res.to_csv(config.REPORTS_DIR / f"{args.out}.csv", index=False)
    lines = ["| model | MAE min (mean ± std) | RMSE min (mean ± std) | per-fold MAE |", "|---|---|---|---|"]
    for r in res.itertuples():
        lines.append(f"| {r.model} | {r.mae_min_mean:.3f} ± {r.mae_min_std:.3f} | {r.rmse_min_mean:.3f} ± {r.rmse_min_std:.3f} | {r.per_fold_mae} |")
    (config.REPORTS_DIR / f"{args.out}.md").write_text(NL.join(lines) + NL, encoding="utf-8")


def row_from_oof(label):
    o = pd.read_parquet(oof_dir / f"{slug(label)}.parquet")
    per_fold = [cv.metrics(g["y"], g["pred"]) for _, g in o.groupby("fold")]
    return {"model": label, **cv.summarize(per_fold), "per_fold_mae": [round(d["mae_min"], 3) for d in per_fold]}


rows = []
for name, factory, variant, iters in EXPERIMENTS:
    if args.only and args.only not in name:
        continue
    checkpoints = iters or [None]
    labels = labels_for(name, iters)
    have_all = all((oof_dir / f"{slug(l)}.parquet").exists() for l in labels)
    if not args.summarize_only and not (args.skip_existing and have_all):
        Xv = X.drop(columns=NO_GEO) if variant == "nogeo" else X
        oof = {c: [] for c in checkpoints}
        t0 = time.time()
        for f in folds:
            m = factory()
            hl = getattr(m, "recency_halflife_days", None)
            w = models.recency_weights(sample["pickup"].iloc[f.train_idx], hl) if hl else None
            m = m.fit(Xv.iloc[f.train_idx], y[f.train_idx], **({"sample_weight": w} if hl else {}))
            for c in checkpoints:
                p = m.predict(Xv.iloc[f.val_idx], num_iteration=c) if c else m.predict(Xv.iloc[f.val_idx])
                oof[c].append(pd.DataFrame({"idx": f.val_idx, "fold": f.k, "y": y[f.val_idx], "pred": p}))
        for c, label in zip(checkpoints, labels):
            pd.concat(oof[c]).to_parquet(oof_dir / f"{slug(label)}.parquet")
        print(f"{name}: fitted in {time.time() - t0:.0f}s", flush=True)
    for label in labels:
        if not (oof_dir / f"{slug(label)}.parquet").exists():
            continue
        r = row_from_oof(label)
        rows.append(r)
        print(f"{label:55s} MAE {r['mae_min_mean']:.3f}±{r['mae_min_std']:.3f}  RMSE {r['rmse_min_mean']:.3f}±{r['rmse_min_std']:.3f}", flush=True)
    write_results(rows)  # written after every experiment so an interrupted run keeps its results
