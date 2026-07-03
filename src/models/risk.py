"""Classic risk metrics over a wide daily-returns DataFrame.

Every function takes returns in WIDE shape (date index, one column per
ticker) — produced by src.warehouse.duck.returns_wide(). A "portfolio"
here is equal-weighted unless weights are passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# why 252: the standard trading-day count per year; used to scale daily
# stats to annual ones (mean scales by T, vol by sqrt(T) under iid returns).
TRADING_DAYS = 252

# why 21: ~one trading month — short enough to react to regime shifts,
# long enough not to be pure noise.
ROLLING_WINDOW = 21

VAR_LEVELS = (0.95, 0.99)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Daily portfolio return. Equal-weight by default — a deliberate neutral
    choice: we are analysing risk structure, not optimising allocation."""
    if weights is None:
        weights = pd.Series(1 / returns.shape[1], index=returns.columns)
    return (returns * weights).sum(axis=1).rename("portfolio")


def annualized_return(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Mean daily return scaled to a year (arithmetic, not compounded —
    fine for comparing names, don't quote it as a realised CAGR)."""
    return returns.mean() * TRADING_DAYS


def annualized_vol(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Daily standard deviation scaled by sqrt(252) — volatility grows with
    the square root of time under independent daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def rolling_vol(returns: pd.DataFrame | pd.Series, window: int = ROLLING_WINDOW) -> pd.DataFrame | pd.Series:
    """Rolling annualized volatility — the 'risk regime' line on the dashboard."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def hist_var(returns: pd.DataFrame | pd.Series, level: float = 0.95) -> pd.Series | float:
    """Historical Value-at-Risk, reported as a POSITIVE loss fraction.

    VaR(95) answers: "on the worst 5% of days, I lose at least this much."
    Historical VaR is just the empirical quantile of past returns — no
    distributional assumption (unlike parametric VaR, which assumes normality
    and famously understates tails).
    """
    if isinstance(returns, pd.Series):
        return -np.quantile(returns.dropna(), 1 - level)
    return returns.apply(lambda col: -np.quantile(col.dropna(), 1 - level))


def max_drawdown(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Worst peak-to-trough loss of the cumulative wealth curve, as a
    NEGATIVE fraction (e.g. -0.34 = lost 34% from the peak).

    Drawdown is the risk number humans actually feel: 'how much of my money
    was gone at the worst point', which variance-based metrics hide.
    """
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.min()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily returns — the raw material for
    the network graph in session 2."""
    return returns.corr()


def summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker metrics plus an equal-weight 'PORTFOLIO' row — the table
    behind the dashboard KPI cards."""
    port = portfolio_returns(returns)
    both = returns.copy()
    both["PORTFOLIO"] = port
    out = pd.DataFrame({
        "ann_return": annualized_return(both),
        "ann_vol": annualized_vol(both),
        "var_95": hist_var(both, 0.95),
        "var_99": hist_var(both, 0.99),
        "max_drawdown": both.apply(max_drawdown),
    })
    return out


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    r = returns_wide()
    print(summary(r).round(4))
