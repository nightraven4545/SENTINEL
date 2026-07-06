"""Mean-variance portfolio optimisation (Markowitz) — the allocation layer.

Everything else in Sentinel *measures* the equal-weight (1/N) book. This asks the
decision question: what SHOULD the weights be? Four classic answers, all
long-only and fully invested:

  • Min-variance  — the lowest-risk portfolio, ignoring returns
  • Max-Sharpe    — the tangency portfolio, best risk-adjusted return
  • Risk-parity   — every name contributes equal risk (the Euler contributions
                    from risk.py, equalised — not equal *weights*, equal *risk*)
  • Equal-weight  — the 1/N baseline the rest of the app analyses

plus the efficient frontier they all live on.

Moments are annualised the same way as the risk module (mean × 252, cov × 252);
optimisation is scipy SLSQP with a fully-invested (Σw = 1) constraint and no
shorting (0 ≤ w ≤ 1) — realistic and keeps every weight interpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.risk import RISK_FREE_RATE, TRADING_DAYS


def annualized_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised expected returns (μ) and covariance (Σ) from daily returns."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_stats(weights, mu, cov, rf: float = RISK_FREE_RATE) -> dict:
    """Annualised return, volatility and Sharpe ratio of a weight vector."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ np.asarray(mu))
    vol = float(np.sqrt(w @ np.asarray(cov) @ w))
    return {"return": ret, "vol": vol, "sharpe": (ret - rf) / vol}


_FULLY_INVESTED = {"type": "eq", "fun": lambda w: w.sum() - 1.0}


def _minimize(objective, n: int, extra_constraints=()) -> np.ndarray:
    """SLSQP over the long-only simplex (0 ≤ w ≤ 1, Σw = 1)."""
    res = minimize(objective, np.repeat(1.0 / n, n), method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[_FULLY_INVESTED, *extra_constraints],
                   options={"maxiter": 1000, "ftol": 1e-12})
    return res.x


def min_variance(returns: pd.DataFrame) -> pd.Series:
    """Weights that minimise portfolio variance (wᵀΣw)."""
    _, cov = annualized_moments(returns)
    C = cov.to_numpy()
    w = _minimize(lambda w: w @ C @ w, C.shape[0])
    return pd.Series(w, index=returns.columns, name="min_variance")


def max_sharpe(returns: pd.DataFrame, rf: float = RISK_FREE_RATE) -> pd.Series:
    """Tangency portfolio — maximises (μᵀw − rf) / √(wᵀΣw)."""
    mu, cov = annualized_moments(returns)
    m, C = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w):
        return -(w @ m - rf) / np.sqrt(w @ C @ w)

    w = _minimize(neg_sharpe, C.shape[0])
    return pd.Series(w, index=returns.columns, name="max_sharpe")


def risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Equal-risk-contribution weights: every name supplies 1/N of portfolio
    variance. Solved by driving each Euler risk share toward 1/N."""
    _, cov = annualized_moments(returns)
    C = cov.to_numpy()
    n = C.shape[0]

    def dispersion(w):
        var = w @ C @ w
        pct_risk = w * (C @ w) / var  # Euler risk shares; sum to 1
        return np.sum((pct_risk - 1.0 / n) ** 2)

    w = _minimize(dispersion, n)
    return pd.Series(w, index=returns.columns, name="risk_parity")


def efficient_frontier(returns: pd.DataFrame, n_points: int = 40) -> pd.DataFrame:
    """Minimum variance at each target return across the achievable range —
    the (vol, return) curve every optimal portfolio sits on."""
    mu, cov = annualized_moments(returns)
    m, C = mu.to_numpy(), cov.to_numpy()
    rows = []
    for target in np.linspace(m.min(), m.max(), n_points):
        hit_target = {"type": "eq", "fun": lambda w, t=target: w @ m - t}
        w = _minimize(lambda w: w @ C @ w, C.shape[0], (hit_target,))
        rows.append({"return": float(w @ m), "vol": float(np.sqrt(w @ C @ w))})
    return pd.DataFrame(rows)


def compare_portfolios(returns: pd.DataFrame,
                       rf: float = RISK_FREE_RATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stats table + weights table for all four portfolios, side by side."""
    mu, cov = annualized_moments(returns)
    ports = {
        "Equal-weight": pd.Series(1.0 / returns.shape[1], index=returns.columns),
        "Min-variance": min_variance(returns),
        "Max-Sharpe": max_sharpe(returns, rf),
        "Risk-parity": risk_parity(returns),
    }
    stats = pd.DataFrame({name: portfolio_stats(w, mu, cov, rf)
                          for name, w in ports.items()}).T
    weights = pd.DataFrame(ports)  # index = tickers, columns = portfolio names
    return stats, weights


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide
    ensure_loaded()
    w = returns_wide()
    stats, weights = compare_portfolios(w)
    pd.set_option("display.width", 200)
    print("Portfolio comparison (annualized):")
    print(stats.round(3))
    print("\nWeights:")
    print(weights.round(3))
