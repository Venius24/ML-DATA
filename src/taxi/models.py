"""Baselines and models. All take the feature frame from ``features.build_features``.

Interface: ``fit(X, y_seconds)`` then ``predict(X) -> seconds``.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from . import config


def recency_weights(pickup: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay: a trip ``half_life_days`` older than the newest training trip gets half the weight."""
    age_days = (pickup.max() - pickup).dt.total_seconds().to_numpy() / 86400.0
    return 0.5 ** (age_days / half_life_days)


class GlobalMedian:
    name = "global median"

    def fit(self, X, y):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class RouteHourMedian:
    """Median duration by (pickup zone, dropoff zone, hour) with a backoff chain.

    A cell is used only if it has at least ``min_count`` training trips; otherwise we back off to
    (pu, do), then (pu borough, do borough, hour), then the global median.
    """

    LEVELS = [["pu", "do", "hour"], ["pu", "do"], ["pu_borough", "do_borough", "hour"]]

    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.name = f"route-hour median (min_count={min_count})"

    def fit(self, X, y):
        d = X[sorted({c for lv in self.LEVELS for c in lv})].copy()
        d["y"] = np.asarray(y)
        self.tables_ = []
        for lv in self.LEVELS:
            g = d.groupby(lv)["y"].agg(["median", "size"]).reset_index()
            g = g[g["size"] >= self.min_count].drop(columns="size")
            self.tables_.append((lv, g.rename(columns={"median": "pred"})))
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        pred = np.full(len(X), np.nan)
        for lv, tab in self.tables_:
            m = X[lv].merge(tab, on=lv, how="left")["pred"].to_numpy()
            pred = np.where(np.isnan(pred), m, pred)
        return np.where(np.isnan(pred), self.global_, pred)


class RidgeModel:
    """Ridge on log(duration) with one-hot zones/time and distance interactions."""

    CAT = ["pu", "do", "pu_borough", "do_borough", "pu_airport", "do_airport", "hour_x_weekend", "dow", "vendor"]

    def __init__(self, alpha: float = 10.0, geo: bool = True, smearing: bool = False):
        self.alpha, self.geo, self.smearing = alpha, geo, smearing
        self.name = f"ridge (alpha={alpha}, geo={geo}, smearing={smearing})"

    def _dense(self, X: pd.DataFrame) -> np.ndarray:
        cols = {
            "same_zone": X["same_zone"],
            "is_holiday": X["is_holiday"],
            "pax_missing": X["passenger_count"].isna().astype(float),
            "pax": X["passenger_count"].fillna(1.0),
            "hour_sin": X["hour_sin"],
            "hour_cos": X["hour_cos"],
        }
        if self.geo:
            # zones 264/265 (unknown) have no polygon: impute 0 distance and flag it
            mi = (X["dist_ft"] / 5280.0).fillna(0.0)
            ld = np.log1p(mi)
            cols.update(
                geo_missing=X["dist_ft"].isna().astype(float),
                log_dist=ld,
                dist_mi=mi,
                dist_mi2=mi**2,
                manh_mi=(X["manhattan_ft"] / 5280.0).fillna(0.0),
                ld_wkend=ld * X["is_weekend"],
                ld_sin=ld * X["hour_sin"],
                ld_cos=ld * X["hour_cos"],
            )
        A = pd.DataFrame(cols).astype("float64").to_numpy()
        return A

    def _prep(self, X, fit: bool):
        C = X[["pu", "do", "pu_borough", "do_borough", "pu_airport", "do_airport", "dow", "vendor"]].copy()
        C["hour_x_weekend"] = X["hour"] + 24 * X["is_weekend"]
        A = self._dense(X)
        if fit:
            self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(C[self.CAT])
            self.mu_, self.sd_ = A.mean(0), A.std(0) + 1e-9
        return sparse.hstack([self.enc_.transform(C[self.CAT]), sparse.csr_matrix((A - self.mu_) / self.sd_)]).tocsr()

    def fit(self, X, y):
        ly = np.log(np.asarray(y))
        M = self._prep(X, fit=True)
        self.model_ = Ridge(alpha=self.alpha).fit(M, ly)
        resid = ly - self.model_.predict(M)
        self.smear_ = float(np.mean(np.exp(resid))) if self.smearing else 1.0  # Duan smearing factor
        return self

    def predict(self, X):
        return np.exp(self.model_.predict(self._prep(X, fit=False))) * self.smear_


LGB_CATEGORICAL = ["pu", "do", "pu_borough", "do_borough", "pu_airport", "do_airport"]
_CAT_LEVELS = {
    "pu": list(range(1, 266)),
    "do": list(range(1, 266)),
    "pu_borough": list(range(0, 8)),
    "do_borough": list(range(0, 8)),
    "pu_airport": [0, 1, 2, 3],
    "do_airport": [0, 1, 2, 3],
}


def to_lgb_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in LGB_CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=_CAT_LEVELS[c])
    return X


class LGBMModel:
    """LightGBM regressor. ``target`` is 'log' (L2 on log seconds) or 'raw' (seconds)."""

    def __init__(self, target="log", objective="regression", n_estimators=600, learning_rate=0.05,
                 num_leaves=127, min_child_samples=50, drop=(), tag="", recency_halflife_days=None, alpha=None):
        self.target, self.objective, self.drop = target, objective, list(drop)
        self.recency_halflife_days = recency_halflife_days  # None = uniform weights
        self.params = dict(
            objective=objective, n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, cat_smooth=20, random_state=config.SEED,
            deterministic=True, force_row_wise=True, verbose=-1, n_jobs=int(os.environ.get("TAXI_THREADS", "0")) or None,
        )
        if alpha is not None:  # quantile level for objective="quantile"
            self.params["alpha"] = alpha
        self.name = f"lgbm {target}/{objective} leaves={num_leaves} lr={learning_rate}{tag}"

    def _frame(self, X):
        return to_lgb_frame(X.drop(columns=self.drop))

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype="float64")
        t = np.log(y) if self.target == "log" else y
        self.model_ = lgb.LGBMRegressor(**self.params).fit(self._frame(X), t, sample_weight=sample_weight)
        return self

    def predict(self, X, num_iteration=None):
        p = self.model_.predict(self._frame(X), num_iteration=num_iteration)
        return np.exp(p) if self.target == "log" else np.maximum(p, 1.0)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model_.booster_.feature_importance("gain"), index=self.model_.booster_.feature_name())
