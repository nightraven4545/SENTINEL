"""Fama-French factor regressions — the multi-factor generalisation of CAPM.

capm() in risk.py explains the portfolio with one factor (the market). This
regresses excess returns on six academic factors (market, size, value,
profitability, investment, momentum) to answer two CFA questions:
  • loadings — what styles is the portfolio actually exposed to?
  • alpha    — what return survives *after* stripping those exposures out?
That factor-adjusted alpha is a far higher bar than CAPM alpha: it only rewards
return the known style premia can't explain.

OLS is hand-rolled with numpy (betas, standard errors, t-stats, R²) to avoid a
statsmodels dependency, matching the project's light-footprint rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.factors import FACTORS, fetch_factors
from src.models.risk import TRADING_DAYS, portfolio_returns


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Ordinary least squares of y on X (X must already include an intercept
    column). Returns coefficients with their standard errors, t-stats and R²."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    sigma2 = ss_res / (n - k)  # unbiased residual variance
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return {"beta": beta, "se": se, "tstat": beta / se,
            "r2": 1 - ss_res / ss_tot, "n": n}


def factor_regression(returns: pd.Series, factors: pd.DataFrame | None = None) -> dict:
    """Regress one return stream's daily EXCESS return on the six factors.

    Returns annualized alpha (the intercept × 252) with its t-stat, each factor
    loading (beta) with its t-stat, R² and the sample size. Aligns on the dates
    the factor library covers (it lags the market by a few weeks).
    """
    factors = fetch_factors() if factors is None else factors
    joined = pd.concat([returns.rename("ret"), factors], axis=1, join="inner").dropna()
    y = (joined["ret"] - joined["rf"]).to_numpy()
    X = np.column_stack([np.ones(len(joined)), joined[FACTORS].to_numpy()])

    res = _ols(y, X)
    names = ["alpha", *FACTORS]
    betas = dict(zip(names, res["beta"]))
    tstats = dict(zip(names, res["tstat"]))
    return {
        "alpha_ann": betas["alpha"] * TRADING_DAYS,
        "alpha_t": tstats["alpha"],
        "betas": {f: betas[f] for f in FACTORS},
        "tstats": {f: tstats[f] for f in FACTORS},
        "r2": res["r2"],
        "n": res["n"],
    }


def portfolio_factor_model(returns_wide: pd.DataFrame,
                           factors: pd.DataFrame | None = None,
                           weights: pd.Series | None = None) -> dict:
    """Factor regression of the (equal-weight by default) portfolio."""
    return factor_regression(portfolio_returns(returns_wide, weights), factors)


def factor_loadings(returns_wide: pd.DataFrame,
                    factors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-name factor betas + annualized alpha + R² — a loadings heatmap of
    which constituents carry which style tilts."""
    factors = fetch_factors() if factors is None else factors
    rows = {}
    for ticker in returns_wide.columns:
        r = factor_regression(returns_wide[ticker], factors)
        rows[ticker] = {**r["betas"], "alpha_ann": r["alpha_ann"], "r2": r["r2"]}
    return pd.DataFrame(rows).T[[*FACTORS, "alpha_ann", "r2"]]


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide
    ensure_loaded()
    w = returns_wide()
    m = portfolio_factor_model(w)
    print(f"Portfolio factor model (n={m['n']}, R²={m['r2']:.2f}, "
          f"alpha {m['alpha_ann']:+.1%} [t={m['alpha_t']:.2f}]):")
    for f in FACTORS:
        print(f"  {f:7} beta {m['betas'][f]:+.3f}  (t={m['tstats'][f]:+.2f})")
    print("\nPer-name loadings:")
    pd.set_option("display.width", 200)
    print(factor_loadings(w).round(3))
