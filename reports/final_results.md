# Final evaluation on 2026-05

Training: 500,000 trips sampled from 2026-04 (seed 42); evaluation: 3,996,039 cleaned trips. Final model chosen by lowest mean CV MAE: `lgbm raw-L2 (+geo, recency half-life 7d) @600 trees`.

| model | MAE (min) | RMSE (min) | mean signed error (min) |
|---|---:|---:|---:|
| global median | 10.013 | 16.049 | -4.62 |
| route-hour median (same sample) | 5.450 | 9.344 | -1.95 |
| route-hour median (all training-month trips) | 4.920 | 8.347 | -1.76 |
| best Ridge: ridge a=10 (+geo, smearing) | 5.112 | 8.191 | -0.60 |
| FINAL: lgbm raw-L2 (+geo, recency half-life 7d) @600 trees | 4.359 | 6.971 | -0.48 |

## Test-month cleaning log

| step | removed | remaining |
|---|---:|---:|
| raw rows | 0 | 4,090,836 |
| missing pickup or dropoff timestamp | 0 | 4,090,836 |
| pickup outside 2026-05 | 14 | 4,090,822 |
| duration < 60s (zero, negative or glitch) | 93,355 | 3,997,467 |
| duration > 3h (multi-hour / multi-day) | 1,428 | 3,996,039 |
| pickup or dropoff zone id not in 1..265 | 0 | 3,996,039 |

## Per-day MAE (minutes)

| day | weekday | trips | final | baseline (route-hour) | mean duration | US holiday |
|---|---|---:|---:|---:|---:|---|
| 2026-05-01 | Friday | 135,553 | 4.33 | 5.59 | 19.46 |  |
| 2026-05-02 | Saturday | 132,592 | 4.00 | 4.60 | 16.80 |  |
| 2026-05-03 | Sunday | 118,425 | 3.90 | 4.76 | 16.51 |  |
| 2026-05-04 | Monday | 103,647 | 4.38 | 5.53 | 19.65 |  |
| 2026-05-05 | Tuesday | 126,408 | 4.18 | 5.29 | 19.06 |  |
| 2026-05-06 | Wednesday | 135,831 | 4.33 | 5.62 | 19.47 |  |
| 2026-05-07 | Thursday | 135,023 | 4.32 | 5.70 | 20.01 |  |
| 2026-05-08 | Friday | 132,193 | 4.27 | 5.46 | 19.37 |  |
| 2026-05-09 | Saturday | 130,575 | 3.86 | 4.65 | 16.89 |  |
| 2026-05-10 | Sunday | 118,582 | 3.66 | 4.73 | 17.25 |  |
| 2026-05-11 | Monday | 120,641 | 4.10 | 5.20 | 19.33 |  |
| 2026-05-12 | Tuesday | 138,380 | 4.59 | 6.01 | 20.30 |  |
| 2026-05-13 | Wednesday | 145,903 | 5.11 | 6.80 | 21.30 |  |
| 2026-05-14 | Thursday | 159,848 | 4.58 | 6.20 | 20.76 |  |
| 2026-05-15 | Friday | 152,300 | 4.80 | 6.25 | 20.70 |  |
| 2026-05-16 | Saturday | 151,570 | 4.64 | 5.78 | 19.13 |  |
| 2026-05-17 | Sunday | 128,869 | 4.34 | 5.15 | 18.74 |  |
| 2026-05-18 | Monday | 121,869 | 4.31 | 5.53 | 19.65 |  |
| 2026-05-19 | Tuesday | 141,657 | 4.84 | 6.26 | 20.29 |  |
| 2026-05-20 | Wednesday | 151,911 | 4.89 | 6.44 | 20.21 |  |
| 2026-05-21 | Thursday | 142,598 | 4.85 | 6.47 | 20.60 |  |
| 2026-05-22 | Friday | 116,441 | 4.32 | 4.74 | 17.86 |  |
| 2026-05-23 | Saturday | 113,725 | 4.32 | 4.64 | 16.14 |  |
| 2026-05-24 | Sunday | 93,382 | 3.69 | 4.78 | 15.80 |  |
| 2026-05-25 | Monday | 76,353 | 4.97 | 5.03 | 15.35 | yes |
| 2026-05-26 | Tuesday | 113,010 | 4.20 | 5.24 | 19.10 |  |
| 2026-05-27 | Wednesday | 126,663 | 4.35 | 4.57 | 17.19 |  |
| 2026-05-28 | Thursday | 139,333 | 4.29 | 5.47 | 19.37 |  |
| 2026-05-29 | Friday | 135,914 | 4.17 | 5.36 | 19.32 |  |
| 2026-05-30 | Saturday | 139,266 | 3.85 | 4.40 | 16.84 |  |
| 2026-05-31 | Sunday | 117,577 | 4.19 | 5.12 | 18.40 |  |
