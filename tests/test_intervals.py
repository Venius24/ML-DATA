import numpy as np
import pytest

from taxi import intervals, models


def test_coverage_and_width():
    y = np.array([1.0, 5.0, 10.0, 20.0])
    lo = np.array([0.0, 6.0, 8.0, 10.0])
    hi = np.array([2.0, 9.0, 12.0, 15.0])
    assert intervals.coverage(y, lo, hi) == 0.5
    assert intervals.mean_width(lo, hi) == pytest.approx(3.5)


def test_pinball_is_asymmetric():
    y = np.array([10.0])
    # under-predicting by 2 costs alpha*2, over-predicting by 2 costs (1-alpha)*2
    assert intervals.pinball(y, np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert intervals.pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.2)


def test_cqr_reaches_target_coverage_on_calibration_rows():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # far too narrow: ~38% coverage
    q = intervals.cqr_correction(y, lo, hi, 0.8)
    assert q > 0
    clo, chi = intervals.apply_correction(lo, hi, q, floor=-np.inf)
    assert intervals.coverage(y, clo, chi) == pytest.approx(0.8, abs=0.005)


def test_cqr_can_shrink_too_wide_intervals():
    y = np.zeros(1000)
    q = intervals.cqr_correction(y, np.full(1000, -5.0), np.full(1000, 5.0), 0.8)
    assert q < 0


def test_apply_correction_respects_floor_and_order():
    lo, hi = intervals.apply_correction(np.array([30.0, 100.0]), np.array([40.0, 200.0]), -20.0, floor=60.0)
    assert (lo >= 60).all() and (hi >= lo).all()


def test_lgbm_quantile_alpha_is_passed():
    m = models.LGBMModel("raw", "quantile", alpha=0.9)
    assert m.params["objective"] == "quantile" and m.params["alpha"] == 0.9
    assert "alpha" not in models.LGBMModel("raw", "regression").params
