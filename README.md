# NYC Taxi Trip Duration

![Python](https://img.shields.io/badge/python-3.13-blue) ![LightGBM](https://img.shields.io/badge/model-LightGBM-green) [![tests](https://github.com/Venius24/ML-DATA/actions/workflows/tests.yml/badge.svg)](https://github.com/Venius24/ML-DATA/actions/workflows/tests.yml) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

Predicts how long a NYC yellow taxi ride will take **using only what is known at pickup** (time, pickup and dropoff zone),
trained on April 2026 and tested once on May 2026, a month the model never saw (~4M trips).

**Result:** LightGBM reaches **4.36 min MAE** on the unseen month: 20% better than a strong route-by-hour median baseline,
and better on all 31 test days.

| model (test month 2026-05) | MAE, min | RMSE, min |
|---|---:|---:|
| global median | 10.01 | 16.05 |
| median by route and hour | 5.45 | 9.34 |
| Ridge (log target, zone geometry) | 5.11 | 8.19 |
| **LightGBM (zone geometry, recency weights)** | **4.36** | **6.97** |

All numbers come from scripts in this repo ([`reports/final_results.md`](reports/final_results.md)); none are typed by hand.

## What makes it trustworthy

- **No leakage.** Trip distance, fare, tips and dropoff time are never used as features. A guard in code (`assert_no_leakage`) and tests enforce this.
- **Honest validation.** Time-based CV inside April with a 3-hour gap between train and validation. May was evaluated exactly once;
  the script refuses to run again. CV predicted 4.20 MAE, the real month gave 4.36 (+3.7%).
- **Reproducible.** Fixed seeds, deterministic LightGBM, pinned versions, one command rebuilds every report.

## Where the model struggles

![MAE by slice](reports/error_analysis_after_fix.png)

- **Long trips**: ~10 min MAE beyond 8 miles; zones and hour cannot see route choice or traffic.
- **Airports** (JFK, LaGuardia): highway congestion we have no data for.
- **Drift**: trips got longer over the month, so models trained on earlier days under-predict. Weighting recent trips helped a little.
- **Holidays**: April has none, so on Memorial Day the model gained only 1% over the baseline.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py   # ~135 MB of public TLC data, not stored in the repo
pytest -q                          # 22 tests
python scripts/run_all.py          # EDA, CV, error analysis, final evaluation, summary
```

## Project layout

```
src/taxi/   data cleaning, features, zone geometry, models, CV
scripts/    one script per step (eda, run_cv, error_analysis, final_eval, ...)
tests/      cleaning rules, leakage guard, CV folds, pipeline smoke
reports/    generated tables and plots
docs/       full methodology
```

## More detail

The full write-up is in [docs/methodology.md](docs/methodology.md): cleaning rules with row counts, EDA findings,
all 26 CV experiments, the error analysis, and the limits of these results.

**Data:** NYC TLC yellow taxi trip records ([source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)).
The data is downloaded by script and not redistributed here; check the TLC terms before reusing it. The code is MIT-licensed ([LICENSE](LICENSE)).
