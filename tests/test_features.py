import numpy as np
import pandas as pd
import pytest

from taxi import config, data, features, zones
from conftest import MONTH


def test_no_leaky_columns(raw_trips, toy_zones):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    X = features.build_features(clean, toy_zones)
    assert not features.FORBIDDEN_COLUMNS.intersection(X.columns)
    assert "duration_s" not in X.columns and "pickup" not in X.columns


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError):
        features.assert_no_leakage(pd.DataFrame({"trip_distance": [1.0], "hour": [1]}))


def test_feature_values(toy_zones):
    df = pd.DataFrame(
        {
            "pickup": pd.to_datetime(["2026-05-25 08:30", "2026-05-26 23:15"]),  # Memorial Day, then Tuesday
            "vendor": [1, 2], "passenger_count": [np.nan, 2.0],
            "pu": [132, 5], "do": [132, 1],
            "duration_s": [600.0, 900.0],
        }
    )
    X = features.build_features(df, toy_zones)
    assert X["is_holiday"].tolist() == [1, 0]
    assert X["hour"].tolist() == [8, 23]
    assert X["dow"].tolist() == [0, 1] and X["is_weekend"].tolist() == [0, 0]
    assert X["same_zone"].tolist() == [1, 0]
    assert X["pu_airport"].tolist() == [2, 0] and X["do_airport"].tolist() == [2, 1]
    assert X["dist_ft"].iloc[0] == 0 and X["dist_ft"].iloc[1] > 0
    assert np.isnan(X["passenger_count"].iloc[0])


def test_unknown_zone_geometry_is_nan_not_error(toy_zones):
    df = pd.DataFrame({"pickup": pd.to_datetime(["2026-04-02 10:00"]), "vendor": [1], "passenger_count": [1.0],
                       "pu": [264], "do": [5], "duration_s": [300.0]})
    X = features.build_features(df, toy_zones)
    assert np.isnan(X["dist_ft"].iloc[0])


def test_features_do_not_depend_on_target(raw_trips, toy_zones):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    a = features.build_features(clean, toy_zones)
    b = features.build_features(clean.assign(duration_s=1.0), toy_zones)
    pd.testing.assert_frame_equal(a, b)


def test_polygon_centroid_square_with_hole():
    outer = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]  # clockwise
    hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]       # counter-clockwise
    pts = np.array(outer + hole, dtype=float)
    cx, cy = zones.polygon_centroid(pts, [0, len(outer)])
    assert cx == pytest.approx(5.0) and cy == pytest.approx(5.0)


@pytest.mark.skipif(not config.ZONE_SHAPES_ZIP.exists(), reason="run scripts/download_data.py first")
def test_real_zones_load():
    z = zones.load_zones()
    assert len(z) == 265 and z.loc[132, "airport"] == 2
    d = np.hypot(z.x[161] - z.x[132], z.y[161] - z.y[132]) / 5280
    assert 10 < d < 15  # Midtown to JFK, miles
