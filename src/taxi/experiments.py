"""Registry of the CV experiments, shared by scripts/run_cv.py and scripts/final_eval.py.

Each entry: (name, factory, feature variant, LightGBM tree-count checkpoints or None).
The CV label of a checkpointed entry is ``"<name> @<n> trees"``.
"""
from __future__ import annotations

from . import features, models

NO_GEO = list(features.GEO_FEATURES)

EXPERIMENTS = [
    ("baseline: global median", lambda: models.GlobalMedian(), "geo", None),
    ("baseline: route-hour median", lambda: models.RouteHourMedian(5), "geo", None),
    ("ridge a=1 (no geo)", lambda: models.RidgeModel(1.0, geo=False), "nogeo", None),
    ("ridge a=10 (no geo)", lambda: models.RidgeModel(10.0, geo=False), "nogeo", None),
    ("ridge a=10 (+geo)", lambda: models.RidgeModel(10.0, geo=True), "geo", None),
    ("ridge a=100 (+geo)", lambda: models.RidgeModel(100.0, geo=True), "geo", None),
    ("ridge a=10 (+geo, smearing)", lambda: models.RidgeModel(10.0, geo=True, smearing=True), "geo", None),
    ("lgbm log-L2 (no geo)", lambda: models.LGBMModel("log", "regression", n_estimators=1000), "nogeo", [300, 600, 1000]),
    ("lgbm log-L2 (+geo)", lambda: models.LGBMModel("log", "regression", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L1 (+geo)", lambda: models.LGBMModel("raw", "regression_l1", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L2 (+geo)", lambda: models.LGBMModel("raw", "regression", n_estimators=1000), "geo", [300, 600, 1000]),
    ("lgbm raw-L2 (+geo, recency half-life 7d)", lambda: models.LGBMModel("raw", "regression", n_estimators=600, recency_halflife_days=7), "geo", [300, 600]),
    ("lgbm raw-L2 (+geo, recency half-life 14d)", lambda: models.LGBMModel("raw", "regression", n_estimators=600, recency_halflife_days=14), "geo", [300, 600]),
    ("lgbm raw-L1 (+geo, leaves=31)", lambda: models.LGBMModel("raw", "regression_l1", n_estimators=1000, num_leaves=31), "geo", [300, 600, 1000]),
]


def labels_for(name: str, iters) -> list[str]:
    return [name] if iters is None else [f"{name} @{c} trees" for c in iters]


def resolve(label: str):
    """Return (model factory, feature variant, n_trees or None) for a CV result label."""
    for name, factory, variant, iters in EXPERIMENTS:
        for lab, c in zip(labels_for(name, iters), iters or [None]):
            if lab == label:
                return factory, variant, c
    raise KeyError(label)
