"""Sanity checks for the GARCH volatility-forecasting module.

We simulate a genuine GARCH(1,1) process so the estimated parameters and the
forecast dynamics have a known ground truth to check against.
"""
import numpy as np
import pandas as pd
import pytest

from scipy.stats import norm

from src.models.volatility import (
    FORECAST_HORIZON,
    _std_t_quantile,
    fit_garch,
    forecast_summary,
    forecast_var,
    forecast_vol,
    garch_vs_ewma_backtest,
)


def simulate_garch(n, omega, alpha, beta, seed=0) -> pd.Series:
    """Hand-rolled GARCH(1,1) path with Gaussian innovations."""
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1 - alpha - beta)  # start at the unconditional variance
    out = np.empty(n)
    for i in range(n):
        r = np.sqrt(sigma2) * rng.normal()
        out[i] = r
        sigma2 = omega + alpha * r ** 2 + beta * sigma2
    return pd.Series(out, index=pd.bdate_range("2010-01-01", periods=n))


@pytest.fixture(scope="module")
def garch_series() -> pd.Series:
    # persistence 0.95, comfortably stationary; daily vol ~ sqrt(1e-5/0.05) ~ 1.4%
    return simulate_garch(4000, omega=1e-5, alpha=0.05, beta=0.90, seed=1)


def test_persistence_is_stationary_and_long_run_vol_defined(garch_series):
    p = fit_garch(garch_series)
    assert 0.0 < p["persistence"] < 1.0          # a stationary GARCH is recovered
    assert p["long_run_vol"] is not None and p["long_run_vol"] > 0
    assert p["alpha"] >= 0 and p["beta"] >= 0     # non-negativity constraints hold


def test_forecast_reverts_toward_long_run():
    # a calm base then a fat shock -> current vol far above long-run; the cone
    # must decay monotonically back down toward the long-run level.
    base = simulate_garch(2000, omega=1e-5, alpha=0.05, beta=0.90, seed=2)
    shocked = pd.concat([base, pd.Series([0.12],  # one 12% day at the end
                                         index=pd.bdate_range("2020-01-01", periods=1))])
    f = forecast_vol(shocked)
    path, lr = f["vol_path"], f["long_run_vol"]
    # the shock enters the forecast (sigma_{T+1}), not the last filtered vol
    # (sigma_T = current_vol, computed before the shock landed).
    assert path[0] > f["current_vol"]                     # shock lifts the forecast
    assert path[0] > path[-1] > lr * 0.9                  # then decays back down
    gaps = [abs(x - lr) for x in path]
    assert all(b <= a + 1e-9 for a, b in zip(gaps, gaps[1:]))  # monotone toward lr


def test_var_term_structure_increases_with_horizon(garch_series):
    var = forecast_var(garch_series, 0.95)["var"]
    horizons = sorted(var)
    values = [var[h] for h in horizons]
    assert all(v > 0 for v in values)
    assert values == sorted(values)              # cumulative VaR grows with horizon
    assert all(b > a for a, b in zip(values, values[1:]))


def test_higher_confidence_gives_larger_var(garch_series):
    v95 = forecast_var(garch_series, 0.95)["var"][1]
    v99 = forecast_var(garch_series, 0.99)["var"][1]
    assert v99 > v95 > 0


def test_std_t_quantile_fat_tail_and_normal_fallback():
    # infinite dof -> exactly the normal quantile
    assert _std_t_quantile(0.95, np.inf) == pytest.approx(abs(norm.ppf(0.05)))
    # finite dof is fatter-tailed: deep in the tail it exceeds the normal
    assert _std_t_quantile(0.99, 5) > abs(norm.ppf(0.01))


def test_backtest_scores_both_models(garch_series):
    bt = garch_vs_ewma_backtest(garch_series, 0.95)
    for model in ("garch", "ewma"):
        s = bt[model]
        assert s["observations"] > 0
        assert 0.0 <= s["breach_rate"] <= 1.0
        assert isinstance(s["passes"], bool)


def test_forecast_summary_shape():
    cols = {c: simulate_garch(1200, 1e-5, 0.05, 0.90, seed=s)
            for s, c in enumerate(["A", "B"])}
    frame = pd.DataFrame(cols)
    out = forecast_summary(frame, horizon=FORECAST_HORIZON)
    assert set(out["series"]) == {"A", "B", "PORTFOLIO"}
    port = out["series"]["PORTFOLIO"]
    assert len(port["vol_path"]) == FORECAST_HORIZON
    assert set(port["var_95"]) == set(out["var_horizons"])
    assert "garch" in out["ewma_vs_garch"] and "ewma" in out["ewma_vs_garch"]
