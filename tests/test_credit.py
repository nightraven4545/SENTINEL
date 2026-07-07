"""Merton distance-to-default: self-consistency + known-direction oracles."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.models.credit import (
    DEFAULT_HORIZON,
    _merton_residuals,
    distance_to_default,
    merton_dd,
)
from src.models.risk import RISK_FREE_RATE


def test_solution_satisfies_both_merton_equations():
    # the strongest oracle: the solved (V, sig_V) must zero both equations
    res = merton_dd(equity_value=100.0, equity_vol=0.4, debt=60.0)
    resid = _merton_residuals([res["asset_value"], res["asset_vol"]],
                              100.0, 0.4, 60.0, RISK_FREE_RATE, DEFAULT_HORIZON)
    assert resid == pytest.approx([0.0, 0.0], abs=1e-6)


def test_asset_vol_is_below_equity_vol():
    # leverage amplifies risk: equity is always more volatile than its assets
    assert merton_dd(100.0, 0.4, 60.0)["asset_vol"] < 0.4


def test_more_debt_shortens_distance_to_default():
    low = merton_dd(100.0, 0.3, 20.0)["distance_to_default"]
    high = merton_dd(100.0, 0.3, 90.0)["distance_to_default"]
    assert high < low


def test_higher_equity_vol_shortens_distance_to_default():
    calm = merton_dd(100.0, 0.2, 60.0)["distance_to_default"]
    wild = merton_dd(100.0, 0.6, 60.0)["distance_to_default"]
    assert wild < calm


def test_default_prob_is_normal_cdf_of_negative_dd():
    res = merton_dd(100.0, 0.35, 55.0)
    assert res["default_prob"] == pytest.approx(norm.cdf(-res["distance_to_default"]))


def test_tiny_leverage_gives_negligible_default_prob():
    res = merton_dd(1000.0, 0.3, 1.0)
    assert res["default_prob"] < 1e-6
    assert res["distance_to_default"] > 5


@pytest.mark.parametrize("args", [
    (np.nan, 0.3, 50.0),   # no market cap
    (100.0, 0.3, 0.0),     # no debt
    (0.0, 0.3, 50.0),      # no equity value
    (100.0, 0.0, 50.0),    # no vol
])
def test_missing_or_nonpositive_inputs_return_nan(args):
    assert np.isnan(merton_dd(*args)["distance_to_default"])


def test_distance_to_default_is_sorted_thinnest_first():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.DataFrame({"SAFE": rng.normal(0, 0.01, 500),
                            "RISKY": rng.normal(0, 0.05, 500)}, index=idx)
    mcap = pd.Series({"SAFE": 1000.0, "RISKY": 100.0})
    debt = pd.Series({"SAFE": 100.0, "RISKY": 90.0})
    out = distance_to_default(returns, mcap, debt)
    assert list(out.index) == ["RISKY", "SAFE"]  # closest to the wall first
    assert out.loc["SAFE", "distance_to_default"] > out.loc["RISKY", "distance_to_default"]
