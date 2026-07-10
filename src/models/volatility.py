"""GARCH conditional-volatility forecasting — the forward-looking upgrade to EWMA.

Sentinel's EWMA vol (risk.ewma_vol, lambda=0.94) is already IGARCH(1,1) with
omega=0 and alpha+beta constrained to 1, so its multi-step forecast is a flat
line — it can nowcast today's regime but not project it. GARCH(1,1) frees those
three parameters:

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

so variance mean-reverts to the long-run level omega/(1-alpha-beta). That gives
the two things EWMA structurally cannot: a **volatility term structure** (vol
rising or falling toward its long-run mean over the next N days) and a
**forecast-VaR** curve across horizons.

Everything consumes the WIDE daily-returns panel from warehouse.duck.returns_wide,
exactly like risk.py, and VaR keeps risk.py's positive-loss-fraction convention.

Fat tails: we fit Student-t innovations, not Gaussian. Equity returns are
leptokurtic; a normal GARCH still understates the tail. One argument (dist="t"),
one estimated degrees-of-freedom parameter, materially better tail VaR.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t as student_t

from src.models.risk import (TRADING_DAYS, ewma_vol, kupiec_lr,
                             portfolio_returns)

# GARCH(1,1) with Student-t innovations is the desk default: enough structure to
# capture vol clustering + fat tails, few enough parameters to estimate stably.
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "t"

# why 21: one trading month — the horizon the tactical overlay rebalances on, so
# the forecast answers "what is vol likely to be by the next rebalance?".
FORECAST_HORIZON = 21

# arch fits far more stably when returns are in percent (~O(1)) than in decimals
# (~O(0.01)); we scale in, then divide every vol/variance back out by the same.
_SCALE = 100.0

# horizons (trading days) reported in the VaR term structure
VAR_HORIZONS = (1, 5, 10, 21)


def _fit(series: pd.Series, dist: str = GARCH_DIST):
    """Fit GARCH(1,1) on a clean return series (scaled to percent). Internal."""
    r = series.dropna() * _SCALE
    model = arch_model(r, mean="Constant", vol="GARCH",
                       p=GARCH_P, q=GARCH_Q, dist=dist, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence/scale chatter, not errors
        return model.fit(disp="off")


def _std_t_quantile(level: float, nu: float) -> float:
    """Lower-tail quantile magnitude of a UNIT-VARIANCE Student-t.

    arch's innovations are standardised (variance 1), but a raw Student-t has
    variance nu/(nu-2), so we rescale the scipy quantile by sqrt((nu-2)/nu).
    Falls back to the normal quantile when nu is not finite (Gaussian fit).
    """
    if not np.isfinite(nu):
        return abs(norm.ppf(1 - level))
    return abs(student_t.ppf(1 - level, nu) * np.sqrt((nu - 2) / nu))


def _annualize(daily_variance_scaled: np.ndarray | float):
    """Scaled daily variance -> annualized volatility in decimals."""
    daily_sigma = np.sqrt(daily_variance_scaled) / _SCALE
    return daily_sigma * np.sqrt(TRADING_DAYS)


def fit_garch(series: pd.Series, dist: str = GARCH_DIST) -> dict[str, float]:
    """Estimated GARCH(1,1) parameters and what they imply about the regime.

    persistence (alpha+beta) is the headline: close to 1 means shocks to vol
    decay slowly (vol is 'sticky'); EWMA forces it to exactly 1. long_run_vol
    is the annualized level variance mean-reverts to — the anchor of every
    multi-step forecast.
    """
    res = _fit(series, dist)
    p = res.params
    omega = float(p["omega"])
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    nu = float(p.get("nu", np.inf))
    persistence = alpha + beta
    # long-run daily variance = omega/(1-persistence); undefined if non-stationary
    lr_var = omega / (1 - persistence) if persistence < 1 else np.nan
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "nu": nu if np.isfinite(nu) else None,
        "persistence": persistence,
        "long_run_vol": float(_annualize(lr_var)) if np.isfinite(lr_var) else None,
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
    }


def forecast_vol(series: pd.Series, horizon: int = FORECAST_HORIZON,
                 dist: str = GARCH_DIST) -> dict:
    """The volatility cone: annualized conditional vol for each of the next
    `horizon` days, plus the current level and the long-run level it reverts to.
    """
    res = _fit(series, dist)
    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var = fc.variance.values[-1]  # scaled %^2, one value per step ahead
    vol_path = _annualize(daily_var)
    current_vol = float(res.conditional_volatility.iloc[-1] / _SCALE * np.sqrt(TRADING_DAYS))
    params = fit_garch(series, dist)
    return {
        "horizon": horizon,
        "current_vol": current_vol,
        "vol_path": [float(v) for v in vol_path],
        "long_run_vol": params["long_run_vol"],
        "persistence": params["persistence"],
    }


def forecast_var(series: pd.Series, level: float = 0.95,
                 horizons: tuple[int, ...] = VAR_HORIZONS,
                 dist: str = GARCH_DIST) -> dict:
    """Forecast-VaR term structure: the CUMULATIVE VaR over the next h days for
    each h, as a positive loss fraction (risk.py convention).

    Under GARCH the h-day-ahead variance is the SUM of the per-step variance
    forecasts (daily returns are ~serially uncorrelated), so the term structure
    is not a naive sqrt(h) fan — it bends as vol mean-reverts. The tail quantile
    is the fitted Student-t's, matching the estimated fat-tailedness.
    """
    res = _fit(series, dist)
    nu = float(res.params.get("nu", np.inf))
    q = _std_t_quantile(level, nu)
    hmax = max(horizons)
    daily_var = res.forecast(horizon=hmax, reindex=False).variance.values[-1]
    cum_var_scaled = np.cumsum(daily_var)  # cumulative scaled variance to each day
    var = {}
    for h in horizons:
        sigma_h = np.sqrt(cum_var_scaled[h - 1]) / _SCALE  # decimal h-day sigma
        var[h] = float(q * sigma_h)
    return {"level": level, "nu": nu if np.isfinite(nu) else None, "var": var}


def garch_vs_ewma_backtest(series: pd.Series, level: float = 0.95,
                           dist: str = GARCH_DIST) -> dict:
    """Which conditional-vol filter is better calibrated: GARCH or EWMA?

    Both produce a one-day-ahead VaR from their filtered conditional vol; we
    shift it one day (today's VaR uses only vol known through yesterday) and
    score each with the same Kupiec test as the rolling-quantile backtest.

    Honest ceiling: GARCH parameters are estimated on the FULL sample (the vol
    STATE is still causal, so this is a filter-quality check, not a true
    walk-forward refit — that would mean thousands of fits). EWMA has no
    parameters to leak, so the comparison is close to apples-to-apples.
    """
    # ponytail: full-sample params (no per-day refit) — upgrade path is an
    # expanding-window refit loop if we ever want a strictly out-of-sample number.
    r = series.dropna()
    res = _fit(r, dist)
    nu = float(res.params.get("nu", np.inf))
    q_t = _std_t_quantile(level, nu)
    z = abs(norm.ppf(1 - level))

    cond_sigma = pd.Series(res.conditional_volatility / _SCALE, index=r.index)
    garch_var = (q_t * cond_sigma).shift(1)                 # positive loss VaR
    ewma_sigma = ewma_vol(r) / np.sqrt(TRADING_DAYS)        # daily decimal
    ewma_var = (z * ewma_sigma).shift(1)

    def _score(var_series: pd.Series) -> dict:
        valid = var_series.notna()
        realized, thresh = r[valid], var_series[valid]
        breaches = int((realized < -thresh).sum())          # loss exceeds VaR
        return kupiec_lr(int(valid.sum()), breaches, level)

    return {"level": level, "garch": _score(garch_var), "ewma": _score(ewma_var)}


def forecast_summary(returns: pd.DataFrame, horizon: int = FORECAST_HORIZON,
                     dist: str = GARCH_DIST) -> dict:
    """Per-ticker + equal-weight PORTFOLIO GARCH forecast — the /forecast payload.

    Mirrors risk.summary's shape: one entry per name plus 'PORTFOLIO', each with
    fitted params, the vol cone, and the 95/99 VaR term structure. The
    EWMA-vs-GARCH backtest is run once on the portfolio (the headline claim).
    """
    port = portfolio_returns(returns)
    cols = {**{c: returns[c] for c in returns.columns}, "PORTFOLIO": port}
    series = {}
    for name, s in cols.items():
        try:
            f = forecast_vol(s, horizon, dist)
            series[name] = {
                "params": fit_garch(s, dist),
                "current_vol": f["current_vol"],
                "long_run_vol": f["long_run_vol"],
                "vol_path": f["vol_path"],
                "var_95": forecast_var(s, 0.95, dist=dist)["var"],
                "var_99": forecast_var(s, 0.99, dist=dist)["var"],
            }
        except Exception as exc:  # a single name failing to converge must not sink the table
            series[name] = {"error": str(exc)}
    return {
        "as_of": str(returns.index.max().date()),
        "horizon": horizon,
        "var_horizons": list(VAR_HORIZONS),
        "series": series,
        "ewma_vs_garch": garch_vs_ewma_backtest(port, 0.95, dist),
    }


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    r = returns_wide()
    port = portfolio_returns(r)
    print("params:", fit_garch(port))
    print("cone:", [round(v, 4) for v in forecast_vol(port)["vol_path"]])
    print("VaR term structure 95%:", forecast_var(port, 0.95)["var"])
    print("backtest:", garch_vs_ewma_backtest(port))
