"""Checks for the Markowitz optimiser, with closed-form oracles where possible.

Two exact cases anchor the solvers:
  • uncorrelated 2-asset min-variance  -> weights ∝ 1/variance
  • uncorrelated 2-asset risk-parity    -> weights ∝ 1/volatility
plus property checks (min-var is lowest vol, max-Sharpe is highest Sharpe,
long-only weights sum to one) on a seeded multivariate sample.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.optimize import (
    compare_portfolios,
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_stats,
    risk_parity,
)

# Two orthogonal (zero-covariance) return streams: var(A) = 4·var(B), so σ_A = 2σ_B.
TWO = pd.DataFrame({"A": [0.02, -0.02, 0.02, -0.02],
                    "B": [0.01, 0.01, -0.01, -0.01]})


def sample_returns(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for i, t in enumerate(["W", "X", "Y", "Z"]):
        cols[t] = rng.normal(0.0004 + i * 0.0002, 0.01 + i * 0.004, n)  # rising vol
    return pd.DataFrame(cols)


def test_min_variance_uncorrelated_weights_inverse_to_variance():
    # 1/var weights: w_A = var_B/(var_A+var_B) = 1/5, w_B = 4/5.
    w = min_variance(TWO)
    assert w["A"] == pytest.approx(0.2, abs=1e-4)
    assert w["B"] == pytest.approx(0.8, abs=1e-4)


def test_risk_parity_uncorrelated_weights_inverse_to_volatility():
    # equal risk contribution with σ_A = 2σ_B -> w_A:w_B = 1:2.
    w = risk_parity(TWO)
    assert w["A"] == pytest.approx(1 / 3, abs=1e-3)
    assert w["B"] == pytest.approx(2 / 3, abs=1e-3)


def test_weights_are_long_only_and_fully_invested():
    for fn in (min_variance, max_sharpe, risk_parity):
        w = fn(sample_returns())
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-6).all()


def test_min_variance_has_lowest_vol_and_max_sharpe_highest_sharpe():
    stats, _ = compare_portfolios(sample_returns())
    assert stats["vol"].idxmin() == "Min-variance"
    assert stats["sharpe"].idxmax() == "Max-Sharpe"


def test_portfolio_stats_sharpe_definition():
    mu = pd.Series({"A": 0.10, "B": 0.20})
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = [0.5, 0.5]
    s = portfolio_stats(w, mu, cov, rf=0.02)
    assert s["return"] == pytest.approx(0.15)
    assert s["vol"] == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))
    assert s["sharpe"] == pytest.approx((0.15 - 0.02) / s["vol"])


def test_efficient_frontier_is_monotonic_and_reaches_min_variance():
    r = sample_returns()
    ef = efficient_frontier(r, n_points=25)
    assert ef["return"].is_monotonic_increasing
    assert (ef["vol"] > 0).all()
    # the frontier's minimum vol matches the dedicated min-variance solve
    mv_vol = compare_portfolios(r)[0].loc["Min-variance", "vol"]
    assert ef["vol"].min() == pytest.approx(mv_vol, rel=1e-2)
