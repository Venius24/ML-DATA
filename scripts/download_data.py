"""Download NYC TLC yellow taxi parquet files and the zone lookup into data/.

Usage: python scripts/download_data.py [YYYY-MM ...]
Defaults to the train month and both test months from src/taxi/config.py.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from taxi import config  # noqa: E402


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main(months: list[str]) -> None:
    for m in months:
        fetch(f"{config.BASE_URL}/trip-data/yellow_tripdata_{m}.parquet", config.trip_file(m))
    fetch(f"{config.BASE_URL}/misc/taxi_zone_lookup.csv", config.ZONE_FILE)
    fetch(f"{config.BASE_URL}/misc/taxi_zones.zip", config.ZONE_SHAPES_ZIP)


if __name__ == "__main__":
    main(sys.argv[1:] or [config.TRAIN_MONTH, config.TEST_MONTH, config.TEST_MONTH_V2])
