import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MONTH = "2026-04"
N_BAD = 5  # rows in the appended frame that cleaning must remove


@pytest.fixture
def raw_trips() -> pd.DataFrame:
    """Synthetic raw TLC-style frame with a known set of bad rows appended."""
    rng = np.random.default_rng(0)
    n = 3000
    start = pd.Timestamp("2026-04-01")
    pickup = start + pd.to_timedelta(rng.integers(0, 30 * 24 * 3600, n), unit="s")
    dur = rng.integers(120, 3000, n)
    df = pd.DataFrame(
        {
            "VendorID": rng.choice([1, 2], n),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(dur, unit="s"),
            "passenger_count": rng.choice([1.0, 2.0, np.nan], n),
            "PULocationID": rng.integers(1, 266, n),
            "DOLocationID": rng.integers(1, 266, n),
        }
    )
    bad = pd.DataFrame(
        {
            "VendorID": [1] * 6,
            "tpep_pickup_datetime": pd.to_datetime(
                ["2026-04-10 10:00:00", "2026-04-10 10:00:00", "2026-04-10 10:00:00", "2026-04-10 10:00:00",
                 "2001-01-01 09:00:00", "2026-04-10 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2026-04-10 10:00:00",  # zero length
                 "2026-04-10 09:59:00",  # negative
                 "2026-04-10 10:00:30",  # 30 s
                 "2026-04-12 10:00:00",  # 2 days
                 "2001-01-01 09:30:00",  # outside month
                 "2026-04-10 10:20:00"]),  # valid row, but passenger_count 9 -> recoded to NaN
            "passenger_count": [1.0, 1.0, 1.0, 1.0, 1.0, 9.0],
            "PULocationID": [10] * 6,
            "DOLocationID": [20] * 6,
        }
    )
    return pd.concat([df, bad], ignore_index=True)


@pytest.fixture
def toy_zones() -> pd.DataFrame:
    z = pd.DataFrame(
        {
            "Borough": "Manhattan", "Zone": "z",
            "x": np.linspace(0, 1e5, 265), "y": np.linspace(0, 5e4, 265),
            "borough_code": 3, "airport": 0,
        },
        index=pd.Index(range(1, 266), name="zone"),
    )
    z.loc[132, "airport"] = 2
    z.loc[1, ["airport", "borough_code"]] = [1, 2]
    z.loc[264:265, ["x", "y"]] = np.nan
    return z
