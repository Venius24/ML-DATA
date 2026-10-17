# Error analysis: `lgbm raw-L2 (+geo, recency half-life 7d) @600 trees`

Computed from out-of-fold CV predictions on the training month (all 5 validation blocks pooled, 423,962 trips). Baseline = `baseline: route-hour median`. The held-out test month is not used. Regenerate with `python scripts/error_analysis.py`.

Overall: MAE 4.21 min, RMSE 6.86 min, mean signed error -0.85 min (baseline MAE 5.49). The longest 10% of trips (true duration >= 35.1 min) account for 29% of absolute error and 60% of squared error.

## Airport vs other

| airport trip | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| 0 | 389,820 | 0.92 | 3.96 | 5.16 | -0.74 | 13.55 | 16.62 |
| 1 | 34,142 | 0.08 | 6.99 | 9.21 | -2.21 | 36.05 | 38.49 |

## Time of day

| period | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| night (0-5) | 33,359 | 0.08 | 3.11 | 4.98 | -0.16 | 12.92 | 15.25 |
| other hours | 262,666 | 0.62 | 4.27 | 5.51 | -0.96 | 14.70 | 18.58 |
| weekday rush (7-9, 16-19) | 127,937 | 0.30 | 4.35 | 5.57 | -0.81 | 14.33 | 18.79 |

## Same-zone trips

| same zone | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| 0 | 408,093 | 0.96 | 4.25 | 5.57 | -0.88 | 14.85 | 18.85 |
| 1 | 15,869 | 0.04 | 3.14 | 3.26 | -0.17 | 4.83 | 6.28 |

## Centroid distance bin (miles, known at pickup)

| distance bin | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| <=0.5 | 36,200 | 0.09 | 2.92 | 3.08 | -0.26 | 5.73 | 6.96 |
| 0.5-1 | 90,669 | 0.21 | 2.88 | 3.18 | -0.44 | 8.13 | 9.11 |
| 1-2 | 130,865 | 0.31 | 3.44 | 4.03 | -0.60 | 12.95 | 13.98 |
| 2-4 | 90,244 | 0.21 | 4.20 | 5.48 | -0.96 | 20.17 | 21.46 |
| 4-8 | 48,042 | 0.11 | 6.11 | 9.20 | -1.53 | 28.58 | 31.58 |
| >8 | 25,500 | 0.06 | 10.16 | 16.67 | -2.87 | 48.97 | 52.97 |

## True duration bin (minutes; selecting on the target)

| true duration | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| 1-5 | 34,014 | 0.08 | 3.06 | 2.90 | 3.02 | 3.83 | 3.65 |
| 5-10 | 96,572 | 0.23 | 2.23 | 2.44 | 1.40 | 7.60 | 7.57 |
| 10-20 | 158,514 | 0.37 | 2.99 | 3.64 | -0.20 | 14.28 | 14.52 |
| 20-40 | 103,608 | 0.24 | 5.45 | 7.21 | -2.62 | 26.00 | 27.13 |
| 40-180 | 31,254 | 0.07 | 13.61 | 21.32 | -9.52 | 52.87 | 58.33 |

## passenger_count missing

| passenger_count missing | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| 0 | 333,394 | 0.79 | 4.33 | 5.51 | -0.88 | 13.57 | 18.23 |
| 1 | 90,568 | 0.21 | 3.73 | 5.40 | -0.77 | 17.02 | 18.91 |

## Validation block (0 = Apr 6-10 ... 4 = Apr 26-30): time drift

| block | trips | share | MAE model (min) | MAE baseline (min) | mean signed error (min) | median true (min) | mean true (min) |
|---|---|---|---|---|---|---|---|
| 0 | 79,515 | 0.19 | 4.23 | 5.46 | 0.17 | 14.00 | 17.69 |
| 1 | 82,839 | 0.20 | 4.05 | 5.40 | -1.28 | 14.22 | 17.97 |
| 2 | 87,935 | 0.21 | 4.18 | 5.42 | -1.47 | 14.42 | 18.33 |
| 3 | 92,540 | 0.22 | 4.36 | 5.72 | -1.22 | 14.85 | 18.96 |
| 4 | 81,133 | 0.19 | 4.20 | 5.40 | -0.35 | 14.57 | 18.86 |

## Worst pickup zones by MAE (>= 500 validation trips)

| pu | zone | trips | MAE (min) | bias (min) |
|---|---|---|---|---|
| 76 | East New York | 637 | 10.38 | -0.93 |
| 61 | Crown Heights North | 746 | 7.94 | -1.41 |
| 132 | JFK Airport | 16,080 | 7.28 | -1.88 |
| 226 | Sunnyside | 609 | 6.66 | -0.97 |
| 70 | East Elmhurst | 948 | 6.61 | -2.82 |
| 37 | Bushwick South | 605 | 6.38 | -1.56 |
| 97 | Fort Greene | 602 | 6.35 | -1.70 |
| 138 | LaGuardia Airport | 11,240 | 6.31 | -2.47 |
