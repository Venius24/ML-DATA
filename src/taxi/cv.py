"""Time-based cross-validation inside the training month.

Expanding window over consecutive day blocks. Validation block k covers a fixed span of days;
the training set is every trip picked up before the block starts MINUS a gap equal to the
maximum allowed trip length, so no training trip is still in progress when validation begins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MAX_DURATION_S


@dataclass
class Fold:
    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def time_folds(pickup: pd.Series, month_start: pd.Timestamp, n_folds: int = 5, block_days: int = 5,
               first_val_day: int = 5) -> list[Fold]:
    """Validation blocks start ``first_val_day`` days into the month (default: days 6-10, 11-15, ...)."""
    gap = pd.Timedelta(seconds=MAX_DURATION_S)
    folds = []
    for k in range(n_folds):
        vs = month_start + pd.Timedelta(days=first_val_day + k * block_days)
        ve = vs + pd.Timedelta(days=block_days)
        train = np.flatnonzero((pickup < vs - gap).to_numpy())
        val = np.flatnonzero(((pickup >= vs) & (pickup < ve)).to_numpy())
        folds.append(Fold(k, train, val, vs, ve))
    return folds


def metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {"mae_min": float(np.mean(np.abs(err)) / 60), "rmse_min": float(np.sqrt(np.mean(err**2)) / 60)}


def summarize(per_fold: list[dict]) -> dict:
    df = pd.DataFrame(per_fold)
    out = {}
    for c in ["mae_min", "rmse_min"]:
        out[f"{c}_mean"] = df[c].mean()
        out[f"{c}_std"] = df[c].std(ddof=1)
    return out
