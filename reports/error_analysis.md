# Error analysis: `lgbm raw-L2 (+geo) @600 trees`

Computed from out-of-fold CV predictions on the training month (all 5 validation blocks pooled, 423,962 trips). Baseline = `baseline: route-hour median`. The held-out test month is not used. Regenerate with `python scripts/error_analysis.py`.

Overall: MAE 4.23 min, RMSE 6.93 min, mean signed error -1.05 min (baseline MAE 5.49). The longest 10% of trips (true duration >= 35.1 min) account for 30% of absolute error and 61% of squared error.

## Airport vs other

| airport trip | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| 0 | 389,820 | 0.92 | 3.97 | 5.16 | -0.90 | 13.55 |
| 1 | 34,142 | 0.08 | 7.15 | 9.21 | -2.79 | 36.05 |

## Time of day

| period | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| night (0-5) | 33,359 | 0.08 | 3.10 | 4.98 | -0.20 | 12.92 |
| other hours | 262,666 | 0.62 | 4.29 | 5.51 | -1.17 | 14.70 |
| weekday rush (7-9, 16-19) | 127,937 | 0.30 | 4.39 | 5.57 | -1.03 | 14.33 |

## Same-zone trips

| same zone | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| 0 | 408,093 | 0.96 | 4.27 | 5.57 | -1.08 | 14.85 |
| 1 | 15,869 | 0.04 | 3.13 | 3.26 | -0.22 | 4.83 |

## Centroid distance bin (miles, known at pickup)

| distance bin | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| <=0.5 | 36,200 | 0.09 | 2.91 | 3.08 | -0.34 | 5.73 |
| 0.5-1 | 90,669 | 0.21 | 2.88 | 3.18 | -0.56 | 8.13 |
| 1-2 | 130,865 | 0.31 | 3.44 | 4.03 | -0.74 | 12.95 |
| 2-4 | 90,244 | 0.21 | 4.23 | 5.48 | -1.18 | 20.17 |
| 4-8 | 48,042 | 0.11 | 6.17 | 9.20 | -1.89 | 28.58 |
| >8 | 25,500 | 0.06 | 10.31 | 16.67 | -3.45 | 48.97 |

## True duration bin (minutes; selecting on the target)

| true duration | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| 1-5 | 34,014 | 0.08 | 3.00 | 2.90 | 2.97 | 3.83 |
| 5-10 | 96,572 | 0.23 | 2.18 | 2.44 | 1.32 | 7.60 |
| 10-20 | 158,514 | 0.37 | 2.97 | 3.64 | -0.33 | 14.28 |
| 20-40 | 103,608 | 0.24 | 5.52 | 7.21 | -2.90 | 26.00 |
| 40-180 | 31,254 | 0.07 | 13.98 | 21.32 | -10.27 | 52.87 |

## passenger_count missing

| passenger_count missing | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) |
|---|---|---|---|---|---|---|
| 0 | 333,394 | 0.79 | 4.36 | 5.51 | -1.09 | 13.57 |
| 1 | 90,568 | 0.21 | 3.74 | 5.40 | -0.93 | 17.02 |

## Worst pickup zones by MAE (>= 500 validation trips)

| pu | zone | trips | MAE (min) | bias (min) |
|---|---|---|---|---|
| 76 | East New York | 637 | 10.57 | -0.93 |
| 61 | Crown Heights North | 746 | 8.02 | -1.46 |
| 132 | JFK Airport | 16,080 | 7.42 | -2.33 |
| 70 | East Elmhurst | 948 | 6.86 | -3.54 |
| 226 | Sunnyside | 609 | 6.57 | -1.28 |
| 138 | LaGuardia Airport | 11,240 | 6.50 | -3.15 |
| 97 | Fort Greene | 602 | 6.38 | -1.82 |
| 37 | Bushwick South | 605 | 6.34 | -1.87 |
