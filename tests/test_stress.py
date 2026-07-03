"""Scenario-math sanity checks for the stress engine."""
import numpy as np
import pandas as pd
import pytest

from src.models.stress import (BASELINE_NAME, SCENARIOS, Scenario,
                               apply_scenario, compare)


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (1000, 3)), columns=["A", "B", "C"]
    )


def test_vol_mult_scales_std_but_not_mean(returns):
    shocked = apply_scenario(returns, Scenario("x", "", vol_mult=2.0))
    for col in returns:
        assert shocked[col].std() == pytest.approx(2 * returns[col].std())
        assert shocked[col].mean() == pytest.approx(returns[col].mean())


def test_daily_drift_shifts_mean_but_not_std(returns):
    shocked = apply_scenario(returns, Scenario("x", "", daily_drift=-0.002))
    for col in returns:
        assert shocked[col].mean() == pytest.approx(returns[col].mean() - 0.002)
        assert shocked[col].std() == pytest.approx(returns[col].std())


def test_ticker_drift_only_hits_named_ticker(returns):
    shocked = apply_scenario(returns, Scenario("x", "", ticker_drift={"A": -0.01}))
    assert shocked["A"].mean() == pytest.approx(returns["A"].mean() - 0.01)
    # not bit-exact: B went through mean + (x - mean) * 1.0
    pd.testing.assert_series_equal(shocked["B"], returns["B"])


def test_unknown_ticker_drift_is_ignored(returns):
    shocked = apply_scenario(returns, Scenario("x", "", ticker_drift={"ZZZ": -0.5}))
    pd.testing.assert_frame_equal(shocked, returns)


def test_crash_scenario_worsens_risk(returns):
    table = compare(returns)
    base = table.loc[BASELINE_NAME]
    crash = table.loc["market_crash_2008_style"]
    assert crash["var_95"] > base["var_95"]
    assert crash["max_drawdown"] < base["max_drawdown"]  # more negative
    assert crash["ann_vol"] > base["ann_vol"]


def test_compare_has_baseline_plus_all_scenarios(returns):
    table = compare(returns)
    assert list(table.index) == [BASELINE_NAME, *SCENARIOS]
