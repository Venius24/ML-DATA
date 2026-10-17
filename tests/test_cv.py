import pandas as pd

from taxi import cv, data
from conftest import MONTH


def test_time_folds_respect_order_and_gap(raw_trips):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    start, _ = data.month_bounds(MONTH)
    folds = cv.time_folds(clean["pickup"], start)
    assert len(folds) == 5
    gap = pd.Timedelta(seconds=data.MAX_DURATION_S)
    for f in folds:
        tr, va = clean["pickup"].iloc[f.train_idx], clean["pickup"].iloc[f.val_idx]
        assert tr.max() < f.val_start - gap
        assert va.min() >= f.val_start and va.max() < f.val_end
        assert len(set(f.train_idx) & set(f.val_idx)) == 0
    assert folds[0].val_end == folds[1].val_start  # validation blocks tile the month


def test_metrics():
    m = cv.metrics([60, 120], [120, 60])
    assert m["mae_min"] == 1.0 and m["rmse_min"] == 1.0
