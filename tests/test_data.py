import pandas as pd

from taxi import data
from conftest import MONTH, N_BAD


def test_cleaning_removes_exactly_the_bad_rows(raw_trips):
    clean, steps = data.clean_trips(raw_trips, MONTH)
    by_name = {s.name: s for s in steps}
    assert len(clean) == len(raw_trips) - N_BAD
    assert by_name["pickup outside 2026-04"].removed == 1
    assert sum(s.removed for s in steps) == len(raw_trips) - len(clean)
    assert steps[-1].remaining == len(clean)
    assert clean["duration_s"].between(data.MIN_DURATION_S, data.MAX_DURATION_S).all()


def test_clean_frame_has_only_allowed_columns(raw_trips):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    assert set(clean.columns) == {"pickup", "vendor", "passenger_count", "pu", "do", "duration_s"}


def test_invalid_passenger_counts_become_nan_not_dropped(raw_trips):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    assert clean["passenger_count"].dropna().between(1, data.MAX_PASSENGERS).all()
    assert len(clean) == len(raw_trips) - N_BAD


def test_output_sorted_by_pickup(raw_trips):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    assert clean["pickup"].is_monotonic_increasing


def test_subsample_is_seeded_and_sorted(raw_trips):
    clean, _ = data.clean_trips(raw_trips, MONTH)
    a, b = data.subsample(clean, 500, seed=1), data.subsample(clean, 500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 500 and a["pickup"].is_monotonic_increasing


def test_zone_id_filter():
    t = pd.Timestamp("2026-04-05 12:00")
    raw = pd.DataFrame(
        {"VendorID": [1, 1], "tpep_pickup_datetime": [t, t],
         "tpep_dropoff_datetime": [t + pd.Timedelta(minutes=10)] * 2,
         "passenger_count": [1.0, 1.0], "PULocationID": [5, 300], "DOLocationID": [6, 7]}
    )
    clean, _ = data.clean_trips(raw, MONTH)
    assert clean["pu"].tolist() == [5]
