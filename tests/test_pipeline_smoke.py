"""End-to-end smoke test on a tiny synthetic sample: clean -> features -> fit -> predict."""
import numpy as np
import pandas as pd
import pytest

from taxi import cv, data, features, models
from conftest import MONTH


def _all_models():
    return [
        models.GlobalMedian(),
        models.RouteHourMedian(2),
        models.RidgeModel(10.0, geo=True, smearing=True),
        models.RidgeModel(10.0, geo=False),
        models.LGBMModel("log", "regression", n_estimators=20, num_leaves=7, min_child_samples=5),
        models.LGBMModel("raw", "regression_l1", n_estimators=20, num_leaves=7, min_child_samples=5),
    ]


@pytest.mark.parametrize("model", _all_models(), ids=lambda m: m.name)
def test_models_fit_predict(model, raw_trips, toy_zones):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    X = features.build_features(clean, toy_zones)
    y = clean["duration_s"].to_numpy()
    start, _ = data.month_bounds(MONTH)
    f = cv.time_folds(clean["pickup"], start)[2]
    model.fit(X.iloc[f.train_idx], y[f.train_idx])
    p = model.predict(X.iloc[f.val_idx])
    assert p.shape == (len(f.val_idx),)
    assert np.isfinite(p).all() and (p > 0).all()


def test_learned_model_beats_global_median_on_structured_data(toy_zones):
    """Duration driven by hour and zone distance must be learnable (guards against a silent no-op fit)."""
    rng = np.random.default_rng(1)
    n = 6000
    t = pd.Timestamp("2026-04-01") + pd.to_timedelta(rng.integers(0, 29 * 86400, n), unit="s")
    df = pd.DataFrame({"pickup": t, "vendor": 1, "passenger_count": 1.0,
                       "pu": rng.integers(1, 260, n), "do": rng.integers(1, 260, n)})
    X = features.build_features(df.assign(duration_s=0.0), toy_zones)
    y = 120 + X["dist_ft"].fillna(0).to_numpy() / 40 + 60 * X["hour"].between(16, 19) + rng.normal(0, 20, n)
    tr, va = np.arange(n) < 4500, np.arange(n) >= 4500
    base = models.GlobalMedian().fit(X[tr], y[tr])
    gbm = models.LGBMModel("raw", "regression", n_estimators=100, num_leaves=15, min_child_samples=10).fit(X[tr], y[tr])
    assert cv.metrics(y[va], gbm.predict(X[va]))["mae_min"] < 0.5 * cv.metrics(y[va], base.predict(X[va]))["mae_min"]
