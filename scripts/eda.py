"""Exploratory analysis of the TRAINING month only. Writes plots and reports/EDA.md.

trip_distance is read here purely to describe data quality; it is never a model feature.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taxi import config, data, zones as zmod

config.REPORTS_DIR.mkdir(exist_ok=True)
month = config.TRAIN_MONTH
raw = data.load_trips(month, data.RAW_COLUMNS + ["trip_distance"])
dur_raw = (raw["tpep_dropoff_datetime"] - raw["tpep_pickup_datetime"]).dt.total_seconds()
clean, steps = data.clean_trips(raw[data.RAW_COLUMNS], month)
zones = zmod.load_zones()

stats = {"month": month, "raw_rows": len(raw)}

# --- missing values -------------------------------------------------------------
miss = raw.isna().sum()
stats["missing"] = {k: int(v) for k, v in miss.items() if v > 0}
stats["passenger_missing_share"] = float(raw["passenger_count"].isna().mean())
_extra = data.load_trips(month, ["passenger_count", "RatecodeID", "store_and_fwd_flag", "congestion_surcharge", "Airport_fee"])
stats["missing_block_identical"] = bool(_extra.isna().nunique(axis=1).eq(1).all())
del _extra

# --- raw duration anomalies --------------------------------------------------------
stats["dur_le_0"] = int((dur_raw <= 0).sum())
stats["dur_lt_60"] = int((dur_raw < 60).sum())
stats["dur_gt_3h"] = int((dur_raw > 3 * 3600).sum())
stats["dur_gt_24h"] = int((dur_raw > 86400).sum())
stats["zero_distance"] = int((raw["trip_distance"] == 0).sum())
ok = (dur_raw >= 60) & (dur_raw <= 3 * 3600)
speed = raw["trip_distance"][ok] / (dur_raw[ok] / 3600)
stats["speed_gt_80mph_among_kept"] = int((speed > 80).sum())
stats["pickup_outside_month"] = int(steps[2].removed)
stats["passenger_zero_or_gt6"] = int((raw["passenger_count"].eq(0) | raw["passenger_count"].gt(6)).sum())

# --- cleaned target -----------------------------------------------------------
d = clean["duration_s"]
stats["clean_rows"] = len(clean)
stats["duration_quantiles_min"] = {str(q): float(d.quantile(q) / 60) for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]}
stats["duration_mean_min"] = float(d.mean() / 60)
stats["duration_std_min"] = float(d.std() / 60)

clean = clean.assign(
    hour=clean["pickup"].dt.hour,
    dow=clean["pickup"].dt.dayofweek,
    weekend=clean["pickup"].dt.dayofweek >= 5,
    pu_air=clean["pu"].map(zones["airport"]),
    do_air=clean["do"].map(zones["airport"]),
)
n_weekday_days = clean.loc[~clean.weekend, "pickup"].dt.date.nunique()
n_weekend_days = clean.loc[clean.weekend, "pickup"].dt.date.nunique()
hourly = clean.groupby(["weekend", "hour"])["duration_s"].agg(["median", "size"]).reset_index()
stats["median_min_by_hour_weekday"] = {int(r.hour): round(r["median"] / 60, 2) for _, r in hourly[~hourly.weekend].iterrows()}
stats["median_min_by_hour_weekend"] = {int(r.hour): round(r["median"] / 60, 2) for _, r in hourly[hourly.weekend].iterrows()}
stats["median_min_by_dow"] = {int(k): round(v / 60, 2) for k, v in clean.groupby("dow")["duration_s"].median().items()}
air = clean["pu_air"].gt(0) | clean["do_air"].gt(0)
stats["airport_share"] = float(air.mean())
stats["airport_median_min"] = float(clean.loc[air, "duration_s"].median() / 60)
stats["non_airport_median_min"] = float(clean.loc[~air, "duration_s"].median() / 60)
stats["same_zone_share"] = float((clean["pu"] == clean["do"]).mean())
stats["same_zone_median_min"] = float(clean.loc[clean["pu"] == clean["do"], "duration_s"].median() / 60)
stats["distinct_routes"] = int(clean.groupby(["pu", "do"]).ngroups)
rc = clean.groupby(["pu", "do"]).size()
stats["routes_with_lt_5_trips_share_of_routes"] = float((rc < 5).mean())

# --- plots ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(np.log10(dur_raw[dur_raw > 0]), bins=120, color="#4c78a8")
for v, lab in [(60, "60 s"), (3 * 3600, "3 h")]:
    ax[0].axvline(np.log10(v), color="crimson", ls="--")
    ax[0].text(np.log10(v), ax[0].get_ylim()[1] * 0.9, " " + lab, color="crimson")
ax[0].set_xlabel("raw duration (log10 seconds)")
ax[0].set_ylabel("trips")
ax[0].set_title("Raw duration (cut lines = cleaning thresholds)")
ax[1].hist(d / 60, bins=np.arange(0, 181, 1), color="#59a14f")
ax[1].set_xlabel("cleaned duration (minutes)")
ax[1].set_title("After cleaning: right-skewed, long tail")
fig.tight_layout()
fig.savefig(config.REPORTS_DIR / "duration_distribution.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
cnt = clean.groupby(["weekend", "hour"]).size().unstack(0)
ax[0].plot(cnt.index, cnt[False] / n_weekday_days, label="weekday (per day)")
ax[0].plot(cnt.index, cnt[True] / n_weekend_days, label="weekend (per day)")
ax[0].set_title("Trips per hour of day")
ax[0].set_xlabel("pickup hour")
ax[0].legend()
for flag, lab in [(False, "weekday"), (True, "weekend")]:
    h = hourly[hourly.weekend == flag]
    ax[1].plot(h["hour"], h["median"] / 60, label=lab)
ax[1].set_title("Median duration by pickup hour (min)")
ax[1].set_xlabel("pickup hour")
ax[1].legend()
dw = clean.groupby("dow")["duration_s"].median() / 60
ax[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dw.values, color="#f28e2b")
ax[2].set_title("Median duration by weekday (min)")
fig.tight_layout()
fig.savefig(config.REPORTS_DIR / "time_patterns.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
samp = raw[ok].sample(100_000, random_state=config.SEED)
ax.hexbin(samp["trip_distance"].clip(upper=40), (dur_raw[samp.index] / 60).clip(upper=180), gridsize=60, bins="log", cmap="viridis")
ax.set_xlabel("trip_distance (miles, clipped at 40) - NOT a model feature")
ax.set_ylabel("duration (min)")
ax.set_title("Distance vs duration after cleaning (100k sample)")
fig.tight_layout()
fig.savefig(config.REPORTS_DIR / "distance_vs_duration.png", dpi=130)
plt.close(fig)

(config.REPORTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=1))
(config.REPORTS_DIR / "cleaning_log.md").write_text(
    "| step | rows removed | rows remaining |\n|---|---:|---:|\n"
    + "\n".join(f"| {s.name} | {s.removed:,} | {s.remaining:,} |" for s in steps) + "\n"
)

q = stats["duration_quantiles_min"]
wd, we = stats["median_min_by_hour_weekday"], stats["median_min_by_hour_weekend"]
peak_h = max(wd, key=wd.get)
low_h = min(wd, key=wd.get)
we_peak = max(we, key=we.get)
md = f"""# EDA - {month} (training month only)

Generated by `python scripts/eda.py`; every number below is computed from the data in that run
(raw values in `reports/eda_stats.json`, cleaning counts in `reports/cleaning_log.md`).
The held-out test month was not opened.

## Size and missing values
- {stats['raw_rows']:,} raw trips. Columns with missing values: {stats['missing']}.
- Missingness is one block of rows: passenger_count is missing for {stats['passenger_missing_share']:.1%} of trips, and
  rate code, store-and-forward flag and the two surcharge fields are missing on exactly the same rows ({stats['missing_block_identical']}). We keep these trips and let the
  model see passenger_count as NaN. A further {stats['passenger_zero_or_gt6']:,} rows have passenger_count 0 or above 6 and are also set to NaN.

## Outliers in the target (duration = dropoff - pickup)
- Non-positive durations: {stats['dur_le_0']:,}. Shorter than 60 s: {stats['dur_lt_60']:,}. Zero-distance trips: {stats['zero_distance']:,}.
- Longer than 3 h: {stats['dur_gt_3h']:,}; longer than 24 h: {stats['dur_gt_24h']:,} (forgotten meters / multi-day records).
- {stats['pickup_outside_month']} rows have a pickup timestamp outside {month} (the file contains stray dates such as the year 2001).
- Among trips that survive the duration filter, {stats['speed_gt_80mph_among_kept']:,} have an implied speed above 80 mph, so some
  noise remains. We do not filter on distance or speed because that would use post-trip information to define the evaluation set.
- Thresholds chosen: keep 60 s <= duration <= 3 h. After cleaning {stats['clean_rows']:,} trips remain.

## Target shape
- Cleaned duration quantiles (minutes): 1% {q['0.01']:.1f}, 10% {q['0.1']:.1f}, 25% {q['0.25']:.1f}, median {q['0.5']:.1f}, 75% {q['0.75']:.1f}, 90% {q['0.9']:.1f}, 99% {q['0.99']:.1f}, 99.9% {q['0.999']:.1f}.
  Mean {stats['duration_mean_min']:.1f} min, std {stats['duration_std_min']:.1f} min: strongly right-skewed, so RMSE will be
  dominated by the long tail while MAE reflects typical trips. Plot: `reports/duration_distribution.png`.

## Time patterns (`reports/time_patterns.png`)
- Weekday median duration is longest at {peak_h}:00 ({wd[peak_h]} min) and shortest at {low_h}:00 ({wd[low_h]} min).
  On weekends the longest median is at {we_peak}:00 ({we[we_peak]} min). The weekday and weekend profiles differ,
  so hour x weekend interactions matter.
- Median duration by weekday (Mon=0): {stats['median_min_by_dow']}.

## Geography
- {stats['distinct_routes']:,} distinct (pickup zone, dropoff zone) routes; {stats['routes_with_lt_5_trips_share_of_routes']:.1%} of them have fewer than 5 trips in the full month,
  so route-level medians are sparse in the tail and need a fallback.
- Airport trips (pickup or dropoff in JFK, LGA or EWR) are {stats['airport_share']:.1%} of trips with a median of {stats['airport_median_min']:.1f} min
  vs {stats['non_airport_median_min']:.1f} min for the rest.
- Same-zone trips are {stats['same_zone_share']:.1%} of trips (median {stats['same_zone_median_min']:.1f} min).

## Consequences for modelling
- Clean at 60 s / 3 h and report everything on the cleaned scale; the cleaned evaluation set is therefore a *filtered* population (see README limits).
- Heavy right tail: compare a log-target model with a direct MAE-objective model.
- Sparse routes: baseline (b) needs a fallback chain; tree models get zone geography as a smoother.
- April 2026 contains no US federal holiday, so an `is_holiday` feature is constant in training and cannot be learned from this month.
"""
(config.REPORTS_DIR / "EDA.md").write_text(md, encoding="utf-8")
print(md)
