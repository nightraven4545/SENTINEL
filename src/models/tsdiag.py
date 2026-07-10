"""Time-series diagnostics — the 'do we validate our assumptions?' layer.

Every risk metric in risk.py quietly assumes returns are stationary and serially
independent (that's what lets us scale vol by sqrt(time) and treat VaR as iid).
This module TESTS those assumptions instead of taking them on faith:

- stationarity  — ADF + KPSS on returns (should be stationary) vs the log-price
                  level (should not). The textbook contrast that justifies why we
                  model returns, not prices.
- autocorrelation — ACF/PACF plus Ljung-Box on returns AND on squared returns.
                  Squared-return autocorrelation is the ARCH effect: it is the
                  statistical justification for the GARCH model in volatility.py.
- decompose     — STL of the realized-vol series (returns themselves have no real
                  seasonality; volatility drifts and clusters, so we decompose that).

Consumes the same WIDE returns panel as everything else; all summaries run on the
equal-weight portfolio.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

from src.models.risk import portfolio_returns, rolling_vol

# 5% is the conventional significance level for all four tests here.
SIG_LEVEL = 0.05
# why 21 lags for Ljung-Box: one trading month — long enough to catch the vol
# clustering that persists for weeks after a shock.
LJUNGBOX_LAGS = 21
# why 40 ACF/PACF lags: two trading months, the usual window for eyeballing
# short-memory structure without drowning in noise at long lags.
ACF_LAGS = 40


def _log_price_proxy(returns: pd.Series) -> pd.Series:
    """A non-stationary 'level' series built from returns alone: the log wealth
    index (cumulative log-returns), which is log-price up to an additive constant.
    Lets us show the returns-vs-level stationarity contrast without needing raw
    prices in the warehouse."""
    return np.log((1 + returns.dropna()).cumprod())


def _adf(x: pd.Series) -> dict:
    """Augmented Dickey-Fuller. H0: a unit root (non-stationary). A small p-value
    REJECTS the unit root, i.e. the series is stationary."""
    stat, p, *_ = adfuller(x, autolag="AIC")
    return {"stat": float(stat), "pvalue": float(p),
            "stationary": bool(p < SIG_LEVEL)}


def _kpss(x: pd.Series) -> dict:
    """KPSS. H0 is the OPPOSITE of ADF: stationarity. A small p-value rejects it.
    Reporting both is standard practice — they cross-check each other."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 'p-value outside table' interpolation note
        stat, p, *_ = kpss(x, regression="c", nlags="auto")
    return {"stat": float(stat), "pvalue": float(p),
            "stationary": bool(p > SIG_LEVEL)}


def stationarity(series: pd.Series) -> dict:
    """ADF + KPSS on the return series and on its log-price level.

    The expected, teachable result: returns are stationary (mean-reverting around
    ~0), the level is not (it wanders) — which is exactly why every model here
    works on returns rather than prices.
    """
    r = series.dropna()
    level = _log_price_proxy(r)
    return {
        "returns": {"adf": _adf(r), "kpss": _kpss(r)},
        "level": {"adf": _adf(level), "kpss": _kpss(level)},
    }


def _ljung_box(x: pd.Series, lags: int) -> dict:
    lb = acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)
    return {"lags": lags, "stat": float(lb["lb_stat"].iloc[-1]),
            "pvalue": float(lb["lb_pvalue"].iloc[-1])}


def autocorrelation(series: pd.Series, nlags: int = ACF_LAGS,
                    lb_lags: int = LJUNGBOX_LAGS) -> dict:
    """ACF/PACF plus Ljung-Box on returns and squared returns.

    Returns are usually white noise (no linear autocorrelation — markets are
    efficient), but SQUARED returns are strongly autocorrelated: big moves follow
    big moves. That squared-return dependence is the ARCH effect, and it is the
    formal reason a GARCH volatility model is warranted (volatility.py).
    """
    r = series.dropna()
    acf_v, acf_ci = acf(r, nlags=nlags, alpha=SIG_LEVEL, fft=True)
    pacf_v, pacf_ci = pacf(r, nlags=nlags, alpha=SIG_LEVEL)
    lb_r = _ljung_box(r, lb_lags)
    lb_sq = _ljung_box(r ** 2, lb_lags)
    return {
        "nlags": nlags,
        "acf": acf_v.tolist(), "acf_confint": acf_ci.tolist(),
        "pacf": pacf_v.tolist(), "pacf_confint": pacf_ci.tolist(),
        "ljung_box_returns": {**lb_r, "white_noise": bool(lb_r["pvalue"] > SIG_LEVEL)},
        "ljung_box_squared": {**lb_sq, "arch_effects": bool(lb_sq["pvalue"] < SIG_LEVEL)},
    }


def decompose(series: pd.Series, period: int = LJUNGBOX_LAGS) -> dict:
    """STL decomposition of the 21-day realized-vol series.

    Returns have no meaningful seasonality, so decomposing them is a null result;
    realized volatility, by contrast, trends and clusters, so its trend/residual
    split is informative. Exploratory — a monthly period is a convention, not a
    claim of true monthly seasonality.
    """
    vol = rolling_vol(series.dropna()).dropna()
    stl = STL(vol.reset_index(drop=True), period=period, robust=True).fit()
    return {
        "period": period,
        "index": [str(d.date()) for d in vol.index],
        "observed": vol.tolist(),
        "trend": stl.trend.tolist(),
        "seasonal": stl.seasonal.tolist(),
        "resid": stl.resid.tolist(),
    }


def tsdiag_summary(returns: pd.DataFrame) -> dict:
    """Portfolio-level diagnostics payload — the /diagnostics response.

    Stationarity verdicts + ACF/PACF + the two Ljung-Box tests, all on the
    equal-weight portfolio. Decomposition is left to the dashboard (its arrays
    are large and chart-only).
    """
    port = portfolio_returns(returns)
    return {
        "as_of": str(returns.index.max().date()),
        "stationarity": stationarity(port),
        "autocorrelation": autocorrelation(port),
    }


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    r = returns_wide()
    s = tsdiag_summary(r)
    st = s["stationarity"]
    print("returns ADF p:", round(st["returns"]["adf"]["pvalue"], 4),
          "stationary:", st["returns"]["adf"]["stationary"])
    print("level   ADF p:", round(st["level"]["adf"]["pvalue"], 4),
          "stationary:", st["level"]["adf"]["stationary"])
    ac = s["autocorrelation"]
    print("Ljung-Box returns:", ac["ljung_box_returns"])
    print("Ljung-Box squared:", ac["ljung_box_squared"])
