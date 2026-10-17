"""Error slices from the out-of-fold CV predictions (training month only).

Usage: python scripts/error_analysis.py [--model "<row name in cv_results.csv>"] [--out error_analysis]
Defaults to the CV row with the lowest mean MAE. Writes reports/<out>.md and a bar chart.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taxi import config, data, features, zones as zmod

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument("--baseline", default="baseline: route-hour median")
parser.add_argument("--results", default="cv_results")
parser.add_argument("--out", default="error_analysis")
args = parser.parse_args()


def slug(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


res = pd.read_csv(config.REPORTS_DIR / f"{args.results}.csv")
best = args.model or res.sort_values("mae_min_mean").iloc[0]["model"]
print("analysing:", best)

zones = zmod.load_zones()
full, _ = data.load_clean(config.TRAIN_MONTH)
sample = data.subsample(full, config.MAX_TRAIN_ROWS)
X = features.build_features(sample, zones, geo=True)


def load_oof(label):
    o = pd.read_parquet(config.DATA_DIR / "oof" / f"{slug(label)}.parquet").set_index("idx").sort_index()
    return o


oof = load_oof(best)
base = load_oof(args.baseline)
d = X.loc[oof.index].copy()
d["y"] = oof["y"].to_numpy()
d["pred"] = oof["pred"].to_numpy()
d["base"] = base.loc[oof.index, "pred"].to_numpy()
d["err"] = d["pred"] - d["y"]
d["aerr"] = d["err"].abs()
d["berr"] = (d["base"] - d["y"]).abs()
d["airport"] = (d.pu_airport > 0) | (d.do_airport > 0)
d["dist_mi"] = d["dist_ft"] / 5280
rush = (d.is_weekend == 0) & (d.hour.between(7, 9) | d.hour.between(16, 19))
d["period"] = np.select([rush, d.hour.between(0, 5)], ["weekday rush (7-9, 16-19)", "night (0-5)"], "other hours")
d["true_len"] = pd.cut(d.y / 60, [0, 5, 10, 20, 40, 180], labels=["1-5", "5-10", "10-20", "20-40", "40-180"])
d["dist_bin"] = pd.cut(d.dist_mi, [-0.01, 0.5, 1, 2, 4, 8, 100], labels=["<=0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"])
d["pax_missing"] = d.passenger_count.isna()
d["same"] = d.same_zone == 1


def md_table(t: pd.DataFrame) -> str:
    t = t.reset_index()
    out = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    for row in t.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:,.0f}" if isinstance(v, (int, np.integer)) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in row) + " |")
    return chr(10).join(out)


def table(col, name):
    g = d.groupby(col, observed=True)
    t = pd.DataFrame({
        "trips": g.size(),
        "share": g.size() / len(d),
        "MAE model (min)": g["aerr"].mean() / 60,
        "MAE baseline (min)": g["berr"].mean() / 60,
        "mean signed error (min)": g["err"].mean() / 60,
        "median true (min)": g["y"].median() / 60,
    })
    t.index.name = name
    return t


slices = {
    "Airport vs other": table("airport", "airport trip"),
    "Time of day": table("period", "period"),
    "Same-zone trips": table("same", "same zone"),
    "Centroid distance bin (miles, known at pickup)": table("dist_bin", "distance bin"),
    "True duration bin (minutes; selecting on the target)": table("true_len", "true duration"),
    "passenger_count missing": table("pax_missing", "passenger_count missing"),
}
name_of = zones["Zone"].to_dict()
g = d.groupby("pu", observed=True)
zt = pd.DataFrame({"trips": g.size(), "MAE (min)": g["aerr"].mean() / 60, "bias (min)": g["err"].mean() / 60})
zt = zt[zt.trips >= 500].sort_values("MAE (min)", ascending=False).head(8)
zt.insert(0, "zone", [name_of.get(i, i) for i in zt.index])

overall = {
    "mae": d.aerr.mean() / 60, "rmse": np.sqrt((d.err**2).mean()) / 60, "bias": d.err.mean() / 60,
    "base_mae": d.berr.mean() / 60,
}
# share of total absolute error from the top 10% of true durations
top = d.y >= d.y.quantile(0.9)
tail_share = d.loc[top, "aerr"].sum() / d.aerr.sum()
sq_tail_share = (d.loc[top, "err"] ** 2).sum() / (d["err"] ** 2).sum()

lines = [
    f"# Error analysis: `{best}`",
    "",
    "Computed from out-of-fold CV predictions on the training month (all 5 validation blocks pooled, "
    f"{len(d):,} trips). Baseline = `{args.baseline}`. The held-out test month is not used. "
    "Regenerate with `python scripts/error_analysis.py`.",
    "",
    f"Overall: MAE {overall['mae']:.2f} min, RMSE {overall['rmse']:.2f} min, mean signed error {overall['bias']:+.2f} min "
    f"(baseline MAE {overall['base_mae']:.2f}). The longest 10% of trips (true duration >= {d.y.quantile(0.9) / 60:.1f} min) "
    f"account for {tail_share:.0%} of absolute error and {sq_tail_share:.0%} of squared error.",
    "",
]
for title, t in slices.items():
    lines += [f"## {title}", "", md_table(t), ""]
lines += ["## Worst pickup zones by MAE (>= 500 validation trips)", "", md_table(zt), ""]
(config.REPORTS_DIR / f"{args.out}.md").write_text("\n".join(lines), encoding="utf-8")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, key, title in [(ax[0], "Time of day", "Time of day"), (ax[1], "Centroid distance bin (miles, known at pickup)", "Distance bin (mi)"), (ax[2], "Airport vs other", "Airport trip")]:
    t = slices[key]
    xs = np.arange(len(t))
    a.bar(xs - 0.2, t["MAE model (min)"], 0.4, label="model")
    a.bar(xs + 0.2, t["MAE baseline (min)"], 0.4, label="route-hour median")
    a.set_xticks(xs)
    a.set_xticklabels([str(i) for i in t.index], rotation=15)
    a.set_title(f"MAE by {title}")
    a.set_ylabel("minutes")
    a.legend()
fig.tight_layout()
fig.savefig(config.REPORTS_DIR / f"{args.out}.png", dpi=130)
print("\n".join(lines))
