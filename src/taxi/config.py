"""Shared constants: months, paths, seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

TRAIN_MONTH = "2026-04"  # used for training + time-based CV
TEST_MONTH = "2026-05"   # frozen holdout, evaluated once in scripts/final_eval.py

SEED = 42
MAX_TRAIN_ROWS = 500_000  # subsample cap (after cleaning) so it runs on a laptop CPU

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"


def trip_file(month: str) -> Path:
    return DATA_DIR / f"yellow_tripdata_{month}.parquet"


ZONE_FILE = DATA_DIR / "taxi_zone_lookup.csv"
ZONE_SHAPES_ZIP = DATA_DIR / "taxi_zones.zip"
