"""Taxi zone metadata: borough, airport flag and polygon centroids.

Centroids come from the official TLC zone shapefile (EPSG:2263, feet). They are static
geography, so using them as features cannot leak anything about a particular trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapefile  # pyshp

from . import config

AIRPORT_IDS = {1: "EWR", 132: "JFK", 138: "LGA"}


def _ring_centroid(points: np.ndarray) -> tuple[float, float, float]:
    x, y = points[:, 0], points[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if area == 0:
        return 0.0, x.mean(), y.mean()
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return area, cx, cy


def polygon_centroid(points: np.ndarray, parts: list[int]) -> tuple[float, float]:
    """Area-weighted centroid over all rings (holes have opposite orientation, so they subtract)."""
    bounds = list(parts) + [len(points)]
    total_a = total_x = total_y = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        a, cx, cy = _ring_centroid(np.asarray(points[lo:hi]))
        total_a += a
        total_x += a * cx
        total_y += a * cy
    if total_a == 0:
        arr = np.asarray(points)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return total_x / total_a, total_y / total_a


def load_zones() -> pd.DataFrame:
    """One row per LocationID with borough, airport flag and centroid x/y (feet)."""
    lookup = pd.read_csv(config.ZONE_FILE).rename(columns={"LocationID": "zone"})
    rows = []
    with shapefile.Reader(str(config.ZONE_SHAPES_ZIP)) as sf:
        fields = [f[0] for f in sf.fields[1:]]
        for sr in sf.shapeRecords():
            rec = dict(zip(fields, sr.record))
            cx, cy = polygon_centroid(np.asarray(sr.shape.points), list(sr.shape.parts))
            rows.append({"zone": int(rec["LocationID"]), "x": cx, "y": cy})
    # a LocationID can appear in several polygons; average them
    geo = pd.DataFrame(rows).groupby("zone", as_index=False).mean()
    zones = lookup.merge(geo, on="zone", how="left")
    zones["borough_code"] = zones["Borough"].fillna("Unknown").astype("category").cat.codes
    zones["airport"] = zones["zone"].map(lambda z: {"EWR": 1, "JFK": 2, "LGA": 3}.get(AIRPORT_IDS.get(z), 0))
    return zones.set_index("zone")[["Borough", "Zone", "x", "y", "borough_code", "airport"]]
