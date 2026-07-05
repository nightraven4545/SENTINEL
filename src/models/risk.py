"""Classic risk metrics over a wide daily-returns DataFrame.

Every function takes returns in WIDE shape (date index, one column per
ticker) — produced by src.warehouse.duck.returns_wide(). A "portfolio"
here is equal-weighted unless weights are passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# why 252: the standard trading-day count per year; used to scale daily
# stats to annual ones (mean scales by T, vol by sqrt(T) under iid returns).
TRADING_DAYS = 252

# why a constant: risk-adjusted ratios need a risk-free rate; ~4% matches the
# 3M T-bill regime of the sample period. One named number beats a live FRED
# dependency for a reproducible analysis (FRED ingestion is future work).
RISK_FREE_RATE = 0.04
_DAILY_RF = RISK_FREE_RATE / TRADING_DAYS

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


def expected_shortfall(returns: pd.DataFrame | pd.Series,
                       level: float = 0.95) -> pd.Series | float:
    """Expected Shortfall (CVaR): the MEAN loss on tail days beyond VaR,
    as a positive fraction.

    VaR answers "how bad is the threshold?"; ES answers "when we're past it,
    how bad on average?". ES is also coherent (subadditive) where VaR is not —
    the reason Basel's FRTB replaced 99% VaR with 97.5% ES for market risk.
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna()
        cutoff = np.quantile(r, 1 - level)
        return -float(r[r <= cutoff].mean())
    return returns.apply(lambda col: expected_shortfall(col, level))


def cornish_fisher_var(returns: pd.DataFrame | pd.Series,
                       level: float = 0.95) -> pd.Series | float:
    """Modified (Cornish-Fisher) parametric VaR, as a positive loss fraction.

    Plain parametric VaR assumes normality; equity returns are negatively
    skewed and fat-tailed, so it understates risk. The Cornish-Fisher
    expansion adjusts the normal quantile for the sample's skewness S and
    excess kurtosis K — the standard CFA/FRM "modified VaR".
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna()
        s, k = float(r.skew()), float(r.kurt())  # pandas .kurt() is EXCESS kurtosis
        z = norm.ppf(1 - level)
        z_cf = (z
                + (z ** 2 - 1) * s / 6
                + (z ** 3 - 3 * z) * k / 24
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        return -float(r.mean() + z_cf * r.std())
    return returns.apply(lambda col: cornish_fisher_var(col, level))


def sharpe_ratio(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Annualized excess return per unit of TOTAL volatility."""
    excess = returns - _DAILY_RF
    return excess.mean() / returns.std() * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Like Sharpe, but the denominator only counts DOWNSIDE deviation —
    investors aren't hurt by upside volatility, so penalizing it (as Sharpe
    does) understates asymmetric strategies."""
    excess = returns - _DAILY_RF
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return excess.mean() * TRADING_DAYS / downside


def calmar_ratio(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Annualized return over worst drawdown — return per unit of the pain
    an investor actually lived through."""
    if isinstance(returns, pd.Series):
        return float(annualized_return(returns) / abs(max_drawdown(returns)))
    return annualized_return(returns) / returns.apply(max_drawdown).abs()


def drawdown_series(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Running drawdown of the wealth curve (0 at peaks, negative underwater) —
    the dashboard's 'underwater' plot."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest underwater stretch in TRADING DAYS (peak to full recovery).

    Depth says how much was lost; duration says how long an investor waited
    to be made whole — often the number that actually breaks conviction.
    """
    underwater = drawdown_series(returns) < 0
    runs = underwater.groupby((~underwater).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def capm(portfolio: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Single-factor CAPM stats vs a benchmark, on EXCESS daily returns.

    beta   — sensitivity to the market factor
    alpha  — annualized return unexplained by beta (the 'skill' term)
    r2     — share of variance the market explains
    tracking_error / information_ratio — active-return risk and its reward
    """
    joined = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    rp, rb = joined.iloc[:, 0] - _DAILY_RF, joined.iloc[:, 1] - _DAILY_RF
    beta = float(rp.cov(rb) / rb.var())
    alpha_ann = float((rp.mean() - beta * rb.mean()) * TRADING_DAYS)
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    return {
        "beta": beta,
        "alpha_ann": alpha_ann,
        "r2": float(joined.iloc[:, 0].corr(joined.iloc[:, 1]) ** 2),
        "tracking_error": te,
        "information_ratio": float(active.mean() * TRADING_DAYS / te),
    }


def risk_contributions(returns: pd.DataFrame,
                       weights: pd.Series | None = None) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility: how much risk each name
    CONTRIBUTES (weight x marginal contribution), not how much it has alone.

    The identity that makes this the desk standard: the contributions sum
    exactly to portfolio vol, so 'pct_of_risk' is a true attribution — a
    10% weight can carry 20% of the risk, and that gap is the finding.
    """
    if weights is None:
        weights = pd.Series(1 / returns.shape[1], index=returns.columns)
    cov = returns.cov() * TRADING_DAYS
    port_vol = float(np.sqrt(weights @ cov @ weights))
    mctr = (cov @ weights) / port_vol
    ctr = weights * mctr
    return pd.DataFrame({
        "weight": weights,
        "risk_contribution": ctr,          # annualized vol units; sums to port vol
        "pct_of_risk": ctr / port_vol,     # sums to 1
    })


# why 250: ~one trading year of history per VaR estimate — the Basel
# minimum observation window for internal models.
VAR_BACKTEST_WINDOW = 250


def kupiec_test(returns: pd.Series, level: float = 0.95,
                window: int = VAR_BACKTEST_WINDOW) -> dict[str, float]:
    """Out-of-sample VaR backtest with Kupiec's proportion-of-failures test.

    Each day's VaR is the empirical quantile of the TRAILING `window` days
    only (shifted one day — no look-ahead), then we count realized breaches.
    Kupiec's likelihood ratio asks: is the observed breach rate statistically
    consistent with the promised one? LR ~ chi2(1); p < 0.05 rejects the model.
    """
    r = returns.dropna()
    var_series = r.rolling(window).quantile(1 - level).shift(1)
    realized = r[var_series.notna()]
    n = len(realized)
    p = 1 - level
    if n == 0:  # fewer than `window`+1 observations: no out-of-sample day to test
        return {"observations": 0, "breaches": 0, "expected_breaches": 0.0,
                "breach_rate": float("nan"), "lr_stat": float("nan"),
                "p_value": float("nan"), "passes": False}
    breaches = int((realized < var_series[var_series.notna()]).sum())
    phat = breaches / n
    # log-likelihoods under promised rate p vs observed rate phat
    ll_null = (n - breaches) * np.log(1 - p) + breaches * np.log(p)
    if breaches in (0, n):  # degenerate observed rate: ll is exact limit 0
        ll_alt = 0.0
    else:
        ll_alt = ((n - breaches) * np.log(1 - phat) + breaches * np.log(phat))
    lr = float(-2 * (ll_null - ll_alt))
    p_value = float(1 - chi2.cdf(lr, df=1))
    return {
        "observations": n,
        "breaches": breaches,
        "expected_breaches": round(p * n, 1),
        "breach_rate": round(phat, 4),
        "lr_stat": round(lr, 3),
        "p_value": round(p_value, 4),
        "passes": bool(p_value > 0.05),
    }


def summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker metrics plus an equal-weight 'PORTFOLIO' row — the table
    behind the dashboard KPI cards and the /metrics endpoint."""
    port = portfolio_returns(returns)
    both = returns.copy()
    both["PORTFOLIO"] = port
    out = pd.DataFrame({
        "ann_return": annualized_return(both),
        "ann_vol": annualized_vol(both),
        "sharpe": sharpe_ratio(both),
        "sortino": sortino_ratio(both),
        "var_95": hist_var(both, 0.95),
        "cf_var_95": cornish_fisher_var(both, 0.95),
        "es_95": expected_shortfall(both, 0.95),
        "var_99": hist_var(both, 0.99),
        "max_drawdown": both.apply(max_drawdown),
        "skew": both.skew(),
        "ex_kurtosis": both.kurt(),
    })
    return out


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    r = returns_wide()
    print(summary(r).round(4))
