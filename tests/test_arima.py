"""Sanity checks for the ARIMA realized-vol forecaster and the EMH finding."""
import numpy as np
import pandas as pd
import pytest

from src.models.arima import (arima_vs_garch_vol, fit_arima, forecast_vol_arima,
                              realized_vol_series, returns_forecastability)


def _sim_garch(n, omega, alpha, beta, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        r = np.sqrt(sigma2) * rng.normal()
        out[i] = r
        sigma2 = omega + alpha * r ** 2 + beta * sigma2
    return pd.Series(out, index=pd.bdate_range("2013-01-01", periods=n))


def test_forecast_shapes_positive_and_bracketed():
    vf = forecast_vol_arima(_sim_garch(1500, 1e-5, 0.08, 0.90, seed=1),
                            horizon=21, level=0.95)
    assert len(vf["forecast"]) == len(vf["lower"]) == len(vf["upper"]) == 21
    assert all(f >= 0 for f in vf["forecast"])                 # vol is non-negative
    assert all(lo >= 0 for lo in vf["lower"])                  # lower floored at 0
    assert all(lo <= f <= up for lo, f, up
               in zip(vf["lower"], vf["forecast"], vf["upper"]))  # PI brackets the point


def test_prediction_interval_does_not_shrink_with_horizon():
    vf = forecast_vol_arima(_sim_garch(1500, 1e-5, 0.08, 0.90, seed=2))
    width = [u - l for u, l in zip(vf["upper"], vf["lower"])]
    assert width[-1] >= width[0]                                # uncertainty accumulates


def test_returns_are_economically_flat():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0003, 0.01, 2000))
    eff = returns_forecastability(r)
    assert eff["economically_flat"] is True                    # white noise -> flat


def test_fit_arima_reports_order_and_resid_check():
    y = realized_vol_series(_sim_garch(1200, 1e-5, 0.08, 0.90, seed=4))
    info = fit_arima(y)
    assert len(info["order"]) == 3
    assert isinstance(info["resid_white_noise"], bool)
    assert info["aic"] == pytest.approx(info["aic"])           # finite, not NaN


def test_arima_and_garch_both_forecast_positive_vol():
    out = arima_vs_garch_vol(_sim_garch(1500, 1e-5, 0.08, 0.90, seed=5), horizon=21)
    assert out["arima_end"] > 0 and out["garch_end"] > 0
