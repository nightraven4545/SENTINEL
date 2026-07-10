"""ARIMA/SARIMAX forecasting — pointed where it actually works.

The honest finding first: an ARIMA fit to daily EQUITY RETURNS produces an almost
flat forecast — often order (0,0,0), and even when a small AR term is fit, the
multi-day forecast drifts less than a fraction of one daily move off the mean.
That is not a bug; it is a hands-on demonstration of market efficiency: there is
no economically exploitable linear structure in returns. So we point ARIMA at the
REALIZED-VOLATILITY series instead, which genuinely trends and clusters, and
report the returns result as a finding.

This gives a second, independent opinion on where volatility is heading — a
statistical-model counterpart to the GARCH forecast in volatility.py.

pmdarima's auto_arima is used when installed; otherwise we pick the order with a
small AIC grid whose differencing order d is fixed up-front by an ADF test (fewer
fits, and statistically motivated). Consumes the same WIDE returns panel.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.models.risk import portfolio_returns, rolling_vol
from src.models.volatility import FORECAST_HORIZON

try:  # optional: auto_arima is nicer but pmdarima can be a fussy build (see requirements)
    import pmdarima as pm
    _HAS_PMDARIMA = True
except Exception:
    _HAS_PMDARIMA = False

ARIMA_MAX_P = 2
ARIMA_MAX_Q = 2
ARIMA_MAX_D = 1
SIG_LEVEL = 0.05
RESID_LB_LAGS = 21


def realized_vol_series(returns: pd.DataFrame | pd.Series) -> pd.Series:
    """The modelling target: annualized 21-day realized vol of the equal-weight
    portfolio. Unlike returns, this series is persistent — so it is forecastable."""
    port = returns if isinstance(returns, pd.Series) else portfolio_returns(returns)
    return rolling_vol(port).dropna().reset_index(drop=True)


def _choose_d(y: pd.Series, max_d: int = ARIMA_MAX_D) -> int:
    """Differencing order from an ADF test: difference until the series is
    stationary (or max_d). Cheaper and more principled than grid-searching d."""
    d, x = 0, y
    while d < max_d and adfuller(x, autolag="AIC")[1] >= SIG_LEVEL:
        x = x.diff().dropna()
        d += 1
    return d


def _pick_order(y: pd.Series, max_p: int, max_q: int, max_d: int) -> tuple[int, int, int]:
    """AIC-best (p,d,q) over a small grid, with d fixed by the ADF test above."""
    if _HAS_PMDARIMA:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tuple(pm.auto_arima(y, max_p=max_p, max_q=max_q, max_d=max_d,
                                       seasonal=False, error_action="ignore",
                                       suppress_warnings=True).order)
    d = _choose_d(y, max_d)
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(y, order=(p, d, q)).fit()
            except Exception:
                continue
            if best is None or res.aic < best[1]:
                best = ((p, d, q), res.aic)
    return best[0] if best else (0, d, 0)


def _fit(y: pd.Series, order: tuple[int, int, int] | None):
    y = pd.Series(y).dropna().reset_index(drop=True)
    order = order or _pick_order(y, ARIMA_MAX_P, ARIMA_MAX_Q, ARIMA_MAX_D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return order, ARIMA(y, order=order).fit()


def fit_arima(y: pd.Series, order: tuple[int, int, int] | None = None) -> dict:
    """Fit ARIMA and report the order, AIC, and a residual white-noise check —
    well-specified residuals (high Ljung-Box p) mean the model captured the
    structure it should have."""
    order, res = _fit(y, order)
    lb = acorr_ljungbox(res.resid, lags=[RESID_LB_LAGS], return_df=True)
    p = float(lb["lb_pvalue"].iloc[-1])
    return {"order": list(order), "aic": float(res.aic),
            "resid_ljung_box_p": p, "resid_white_noise": bool(p > SIG_LEVEL)}


def forecast_vol_arima(returns: pd.DataFrame | pd.Series,
                       horizon: int = FORECAST_HORIZON, level: float = 0.95) -> dict:
    """Forecast the realized-vol series with prediction intervals.

    Vol is non-negative, so the lower PI band is floored at 0 (an ARIMA on a
    level series has Gaussian intervals that can otherwise dip below zero).
    """
    y = realized_vol_series(returns)
    order, res = _fit(y, None)
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=1 - level)
    lb = acorr_ljungbox(res.resid, lags=[RESID_LB_LAGS], return_df=True)
    return {
        "horizon": horizon, "order": list(order), "aic": float(res.aic),
        "level": level,
        "resid_white_noise": bool(float(lb["lb_pvalue"].iloc[-1]) > SIG_LEVEL),
        "current_vol": float(y.iloc[-1]),
        "forecast": [float(v) for v in mean],
        "lower": [max(0.0, float(x)) for x in ci.iloc[:, 0]],
        "upper": [float(x) for x in ci.iloc[:, 1]],
    }


def returns_forecastability(returns: pd.DataFrame | pd.Series) -> dict:
    """The EMH finding: fit ARIMA to daily returns and show the forecast is flat.

    An efficient market leaves no economically useful linear signal for ARIMA.
    Even when the AIC grid fits a small AR term (real indices show mild
    autocorrelation from lead-lag effects), the multi-day forecast barely moves:
    we quantify 'barely' as the 5-day forecast spread being under HALF a single
    day's standard deviation — i.e. a week of forecast drift is dwarfed by one
    day of noise.
    """
    port = returns if isinstance(returns, pd.Series) else portfolio_returns(returns)
    order, res = _fit(port, None)
    fc = res.get_forecast(steps=5).predicted_mean
    spread = float(np.max(fc) - np.min(fc))
    daily_sd = float(port.std())
    return {
        "best_order": list(order),
        "mean_daily_return": float(port.mean()),
        "forecast_spread_5d": spread,
        "daily_sd": daily_sd,
        # economically flat: a week of forecast drift < half a single day's move
        "economically_flat": bool(spread < 0.5 * daily_sd),
    }


def arima_vs_garch_vol(returns: pd.DataFrame | pd.Series,
                       horizon: int = FORECAST_HORIZON) -> dict:
    """Two independent reads on 21-day-ahead vol: statistical (ARIMA) vs
    econometric (GARCH). Agreement is reassuring; a gap is itself a signal."""
    vf = forecast_vol_arima(returns, horizon)
    from src.models.volatility import forecast_vol as garch_forecast
    port = returns if isinstance(returns, pd.Series) else portfolio_returns(returns)
    g = garch_forecast(port, horizon)
    return {"horizon": horizon, "current_vol": vf["current_vol"],
            "arima_end": vf["forecast"][-1], "garch_end": g["vol_path"][-1],
            "arima_order": vf["order"]}


def arima_summary(returns: pd.DataFrame, horizon: int = FORECAST_HORIZON,
                  level: float = 0.95) -> dict:
    """The /arima payload: the vol forecast with intervals, the returns-are-
    unforecastable finding, and the ARIMA-vs-GARCH comparison."""
    vf = forecast_vol_arima(returns, horizon, level)
    from src.models.volatility import forecast_vol as garch_forecast
    g = garch_forecast(portfolio_returns(returns), horizon)
    return {
        "as_of": str(returns.index.max().date()),
        "vol_forecast": vf,
        "returns_efficiency": returns_forecastability(returns),
        "vs_garch": {"horizon": horizon, "current_vol": vf["current_vol"],
                     "arima_end": vf["forecast"][-1], "garch_end": g["vol_path"][-1]},
    }


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    r = returns_wide()
    vf = forecast_vol_arima(r)
    print("ARIMA vol order:", vf["order"], "resid white noise:", vf["resid_white_noise"])
    print("current vol:", round(vf["current_vol"], 4),
          "-> 21d forecast:", round(vf["forecast"][-1], 4),
          f"[{round(vf['lower'][-1], 4)}, {round(vf['upper'][-1], 4)}]")
    print("returns efficiency:", returns_forecastability(r))
    print("vs GARCH:", arima_vs_garch_vol(r))
