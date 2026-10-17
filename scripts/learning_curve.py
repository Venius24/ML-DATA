"""Does more training data help? Time-based CV of the selected LightGBM config at several sample sizes.

Same fold definition as run_cv.py; validation rows are whatever falls in each validation block of the
(sub)sample, so metrics are unbiased estimates of the full-month validation error. Training month only.
Writes reports/learning_curve.md.

Usage: python scripts/learning_curve.py [size ...]   (default: 125000 500000 2000000 all)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from taxi import config, cv, data, experiments, features, models, zones as zmod

NL = chr(10)
LABEL = "lgbm raw-L2 (+geo, recency half-life 7d) @600 trees"
sizes = sys.argv[1:] or ["125000", "500000", "2000000", "all"]

zones = zmod.load_zones()
full, _ = data.load_clean(config.TRAIN_MONTH)
month_start, _ = data.month_bounds(config.TRAIN_MONTH)
factory, variant, n_trees = experiments.resolve(LABEL)

rows = []
for sz in sizes:
    n = len(full) if sz == "all" else int(sz)
    sample = data.subsample(full, n)
    X = features.build_features(sample, zones)
    y = sample["duration_s"].to_numpy("float64")
    t0 = time.time()
    per_fold = []
    for f in cv.time_folds(sample["pickup"], month_start):
        m = factory()
        m.params["n_estimators"] = n_trees
        w = models.recency_weights(sample["pickup"].iloc[f.train_idx], m.recency_halflife_days)
        m.fit(X.iloc[f.train_idx], y[f.train_idx], sample_weight=w)
        per_fold.append(cv.metrics(y[f.val_idx], m.predict(X.iloc[f.val_idx])))
    s = cv.summarize(per_fold)
    rows.append({"train sample size": f"{len(sample):,}", **{k: round(v, 3) for k, v in s.items()}, "fit+predict seconds": round(time.time() - t0)})
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
lines = [f"CV of `{LABEL}` by training-sample size (training month only).", "",
         "| sample size | MAE min (mean ± std) | RMSE min (mean ± std) | seconds (5 folds) |", "|---|---|---|---|"]
for r in res.itertuples():
    lines.append(f"| {r[1]} | {r.mae_min_mean:.3f} ± {r.mae_min_std:.3f} | {r.rmse_min_mean:.3f} ± {r.rmse_min_std:.3f} | {r[-1]} |")
(config.REPORTS_DIR / "learning_curve.md").write_text(NL.join(lines) + NL, encoding="utf-8")
