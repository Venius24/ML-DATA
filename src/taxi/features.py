"""Feature builders. Only information available at pickup time is used.

Allowed inputs: pickup timestamp, pickup/dropoff zone (the rider states the destination
when hailing), vendor, passenger count, and static zone geography.
Forbidden: dropoff time, trip_distance, fare/tip/tolls/total, payment type, rate code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

FORBIDDEN_COLUMNS = {
    "tpep_dropoff_datetime", "dropoff", "trip_distance", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount",
    "congestion_surcharge", "Airport_fee", "cbd_congestion_fee", "payment_type",
    "RatecodeID", "store_and_fwd_flag", "duration_s",
}

TIME_FEATURES = ["hour", "tod", "dow", "is_weekend", "is_holiday", "hour_sin", "hour_cos"]
ZONE_FEATURES = ["pu", "do", "pu_borough", "do_borough", "pu_airport", "do_airport", "same_zone"]
TRIP_FEATURES = ["vendor", "passenger_count"]
GEO_FEATURES = ["dist_ft", "manhattan_ft", "dx_ft", "dy_ft"]
CATEGORICAL = ["pu", "do", "pu_borough", "do_borough", "pu_airport", "do_airport"]


def us_holiday_dates(years: list[int]) -> set:
    return set(holidays.US(years=years).keys())


def build_features(df: pd.DataFrame, zones: pd.DataFrame, geo: bool = True) -> pd.DataFrame:
    """Return a feature frame aligned with ``df``; the target is never included."""
    t = df["pickup"]
    out = pd.DataFrame(index=df.index)
    out["hour"] = t.dt.hour.astype("int8")
    out["tod"] = (t.dt.hour + t.dt.minute / 60.0).astype("float32")
    out["dow"] = t.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    hol = us_holiday_dates(sorted(t.dt.year.unique().tolist()))
    out["is_holiday"] = t.dt.date.isin(hol).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["tod"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["tod"] / 24).astype("float32")

    out["pu"] = df["pu"].astype("int16")
    out["do"] = df["do"].astype("int16")
    out["pu_borough"] = df["pu"].map(zones["borough_code"]).astype("int8")
    out["do_borough"] = df["do"].map(zones["borough_code"]).astype("int8")
    out["pu_airport"] = df["pu"].map(zones["airport"]).astype("int8")
    out["do_airport"] = df["do"].map(zones["airport"]).astype("int8")
    out["same_zone"] = (df["pu"] == df["do"]).astype("int8")

    out["vendor"] = df["vendor"].astype("int16")
    out["passenger_count"] = df["passenger_count"].astype("float32")

    if geo:
        px, py = df["pu"].map(zones["x"]), df["pu"].map(zones["y"])
        dx, dy = df["do"].map(zones["x"]), df["do"].map(zones["y"])
        out["dx_ft"] = (dx - px).astype("float32")
        out["dy_ft"] = (dy - py).astype("float32")
        out["dist_ft"] = np.hypot(out["dx_ft"], out["dy_ft"]).astype("float32")
        out["manhattan_ft"] = (out["dx_ft"].abs() + out["dy_ft"].abs()).astype("float32")
    assert_no_leakage(out)
    return out


def assert_no_leakage(X: pd.DataFrame) -> None:
    bad = FORBIDDEN_COLUMNS.intersection(X.columns)
    if bad:
        raise ValueError(f"leaky columns in feature frame: {sorted(bad)}")
