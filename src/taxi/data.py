"""Loading and cleaning of TLC yellow taxi trip records.

The cleaned frame deliberately keeps ONLY columns that are known at pickup plus the
target (``duration_s``). Fare, tip, distance, payment and rate-code columns never leave
this module, which is the first line of defence against leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config

RAW_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
]

MIN_DURATION_S = 60          # shorter trips are meter glitches / cancelled rides
MAX_DURATION_S = 3 * 3600    # longer trips are forgotten meters or multi-hour outliers
VALID_ZONE_IDS = range(1, 266)  # 1..265 in taxi_zone_lookup.csv (264/265 = unknown)
MAX_PASSENGERS = 6


@dataclass
class CleaningStep:
    name: str
    removed: int
    remaining: int


def load_trips(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.trip_file(month), columns=columns or RAW_COLUMNS)


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthBegin(1)


def clean_trips(raw: pd.DataFrame, month: str) -> tuple[pd.DataFrame, list[CleaningStep]]:
    """Apply the documented cleaning rules and return (clean frame, row-count log)."""
    steps: list[CleaningStep] = []
    df = raw
    steps.append(CleaningStep("raw rows", 0, len(df)))

    def apply(name: str, keep: pd.Series) -> None:
        nonlocal df
        before = len(df)
        df = df[keep]
        steps.append(CleaningStep(name, before - len(df), len(df)))

    apply(
        "missing pickup or dropoff timestamp",
        df["tpep_pickup_datetime"].notna() & df["tpep_dropoff_datetime"].notna(),
    )
    start, end = month_bounds(month)
    apply(
        f"pickup outside {month}",
        (df["tpep_pickup_datetime"] >= start) & (df["tpep_pickup_datetime"] < end),
    )
    duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    apply(f"duration < {MIN_DURATION_S}s (zero, negative or glitch)", duration >= MIN_DURATION_S)
    duration = duration.loc[df.index]
    apply(f"duration > {MAX_DURATION_S // 3600}h (multi-hour / multi-day)", duration <= MAX_DURATION_S)
    duration = duration.loc[df.index]
    apply(
        "pickup or dropoff zone id not in 1..265",
        df["PULocationID"].isin(VALID_ZONE_IDS) & df["DOLocationID"].isin(VALID_ZONE_IDS),
    )

    out = pd.DataFrame(
        {
            "pickup": df["tpep_pickup_datetime"],
            "vendor": df["VendorID"].astype("int16"),
            "passenger_count": df["passenger_count"].where(
                df["passenger_count"].between(1, MAX_PASSENGERS)
            ),  # 0 and >6 are recoded to NaN (kept, not dropped)
            "pu": df["PULocationID"].astype("int16"),
            "do": df["DOLocationID"].astype("int16"),
            "duration_s": duration.loc[df.index].astype("float32"),
        }
    ).sort_values("pickup", kind="stable").reset_index(drop=True)
    return out, steps


def subsample(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Uniform random subsample, re-sorted by pickup time (keeps the timeline intact)."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).sort_values("pickup", kind="stable").reset_index(drop=True)


def load_clean(month: str, max_rows: int | None = None):
    clean, steps = clean_trips(load_trips(month), month)
    if max_rows is not None:
        steps.append(CleaningStep(f"random subsample to {max_rows:,} rows (seed {config.SEED})",
                                  max(len(clean) - max_rows, 0), min(len(clean), max_rows)))
        clean = subsample(clean, max_rows)
    return clean, steps
