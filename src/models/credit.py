"""Merton (1974) structural credit risk: distance-to-default from equity.

The CFA/CA crossover of Sentinel. The forensic lens (forensic.py) scores
distress from the *accounting* — Altman's Z off the balance sheet. This scores
the same distress from the *market*, treating a firm's equity as a call option
on its assets. When the two disagree — a safe Altman zone but a thin
distance-to-default, or the reverse — that gap is the finding.

Merton models equity E as a European call on asset value V, struck at the face
value of debt D, maturing in T years:

    E      = V * N(d1) - D * exp(-r*T) * N(d2)
    sig_E * E = N(d1) * sig_V * V              (Ito: equity vol from asset vol)

Two equations, two unknowns (V, sig_V), solved numerically from the observables
(E, sig_E, D). Distance-to-default is then how many asset-vol standard
deviations the asset value sits above the default point, and the (risk-neutral)
default probability is N(-DD).

Inputs come from both lenses: E = shares * price (market), sig_E = annualized
equity vol (market), D = a KMV default point off the balance sheet (accounting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

from src.models.risk import RISK_FREE_RATE, TRADING_DAYS, annualized_vol

# why 1 year: the standard Merton/KMV horizon — default is measured over the
# next 12 months, matching how credit desks quote a 1-year PD.
DEFAULT_HORIZON = 1.0


def _merton_residuals(x, E, sig_E, D, rf, T):
    """The two Merton equations as residuals, for the root finder. x = (V, sig_V)."""
    V, sig_V = x
    # guard the solver's search path from invalid regions (log/÷ of non-positive)
    if V <= 0 or sig_V <= 0:
        return [1e6, 1e6]
    d1 = (np.log(V / D) + (rf + 0.5 * sig_V ** 2) * T) / (sig_V * np.sqrt(T))
    d2 = d1 - sig_V * np.sqrt(T)
    eq1 = V * norm.cdf(d1) - D * np.exp(-rf * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sig_V * V - sig_E * E
    return [eq1, eq2]


def merton_dd(equity_value: float, equity_vol: float, debt: float,
              rf: float = RISK_FREE_RATE, T: float = DEFAULT_HORIZON) -> dict:
    """Solve the Merton model for one firm and return its distance-to-default.

    equity_value  E     — market value of equity (shares * price)
    equity_vol    sig_E — annualized equity volatility
    debt          D     — face value of debt / default point

    Returns asset value & vol (the solved latent firm variables), the
    distance-to-default (risk-neutral, DD = d2), the implied 1-year default
    probability N(-DD), and leverage D/V. NaNs in, NaNs out — a name missing a
    default point (e.g. a bank with an unclassified balance sheet) is not forced.
    """
    nan_out = {"asset_value": np.nan, "asset_vol": np.nan,
               "distance_to_default": np.nan, "default_prob": np.nan,
               "leverage": np.nan}
    if not all(np.isfinite([equity_value, equity_vol, debt])) or \
            equity_value <= 0 or equity_vol <= 0 or debt <= 0:
        return nan_out

    # initial guess: assets ~ equity + PV(debt); asset vol scaled down by leverage
    v0 = equity_value + debt * np.exp(-rf * T)
    s0 = equity_vol * equity_value / v0
    V, sig_V = fsolve(_merton_residuals, [v0, s0],
                      args=(equity_value, equity_vol, debt, rf, T), full_output=False)
    if V <= 0 or sig_V <= 0:  # solver wandered off — report honestly, don't fake it
        return nan_out

    # DD = d2: asset-vol standard deviations from the default point (risk-neutral
    # drift rf — avoids needing an equity-risk-premium estimate for the physical drift).
    dd = (np.log(V / debt) + (rf - 0.5 * sig_V ** 2) * T) / (sig_V * np.sqrt(T))
    return {
        "asset_value": float(V),
        "asset_vol": float(sig_V),
        "distance_to_default": float(dd),
        "default_prob": float(norm.cdf(-dd)),
        "leverage": float(debt / V),
    }


def default_point(fundamentals: pd.DataFrame | None = None) -> pd.Series:
    """KMV default point per ticker: current liabilities + 0.5 * long-term debt.

    The firm rarely defaults at total liabilities: short-term obligations are due
    in full but long-term debt only partially bites within a year. Moody's KMV's
    'short-term debt + half of long-term debt' is the standard barrier; we use
    current liabilities as the short-term leg (the tag most filers report).
    Names lacking either tag (banks' unclassified balance sheets) yield NaN.
    """
    from src.ingest.edgar import latest_fundamentals
    from src.models.fundamentals import _col
    w = latest_fundamentals() if fundamentals is None else fundamentals
    return (_col(w, "current_liabilities") + 0.5 * _col(w, "long_term_debt")).rename(
        "default_point")


def distance_to_default(returns: pd.DataFrame, market_cap: pd.Series,
                        debt: pd.Series | None = None,
                        rf: float = RISK_FREE_RATE,
                        T: float = DEFAULT_HORIZON) -> pd.DataFrame:
    """Merton distance-to-default for every name with the inputs to compute it.

    returns     — wide daily returns (equity vol comes from here)
    market_cap  — market value of equity per ticker (shares * latest close)
    debt        — default point per ticker; defaults to default_point()

    One row per ticker, sorted thinnest cushion first (smallest DtD = closest to
    the wall). The universe wrapper that the dashboard and agent read.
    """
    if debt is None:
        debt = default_point()
    eq_vol = annualized_vol(returns)  # per-ticker annualized equity vol
    rows = {}
    for tkr in market_cap.index:
        if tkr not in returns.columns:
            continue
        rows[tkr] = merton_dd(float(market_cap.get(tkr, np.nan)),
                              float(eq_vol.get(tkr, np.nan)),
                              float(debt.get(tkr, np.nan)), rf, T)
    out = pd.DataFrame(rows).T
    return out.sort_values("distance_to_default")


if __name__ == "__main__":
    from src.ingest.edgar import latest_fundamentals
    from src.ingest.market import fetch_prices
    from src.warehouse.duck import returns_wide

    close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
    shares = latest_fundamentals().get("shares")
    mcap = (shares * close).dropna()
    pd.set_option("display.width", 200)
    print(distance_to_default(returns_wide(), mcap).round(3))
