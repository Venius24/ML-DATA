"""Prediction intervals: metrics and conformal calibration (CQR).

Intervals come from two LightGBM quantile models (lower and upper quantile). Raw quantile models often
miss their nominal coverage, so we optionally widen or narrow both ends by one constant ``q`` learned on
out-of-fold predictions (conformalized quantile regression, Romano et al. 2019).
"""
from __future__ import annotations

import numpy as np


def coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def pinball(y, pred, alpha: float) -> float:
    d = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def cqr_correction(y, lo, hi, target: float) -> float:
    """Constant ``q`` such that [lo - q, hi + q] covers ``target`` of the given (calibration) rows."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(np.ceil((n + 1) * target) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def apply_correction(lo, hi, q: float, floor: float = 0.0):
    lo, hi = np.asarray(lo) - q, np.asarray(hi) + q
    lo = np.maximum(lo, floor)
    return lo, np.maximum(hi, lo)
