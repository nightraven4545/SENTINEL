"""Scenario stress-testing engine.

A Scenario is a parameterized shock applied to the HISTORICAL return series
(historical-simulation stress testing): we replay the past as if the shock
regime were in force, then recompute the portfolio risk metrics. Two knobs:

- vol_mult   — scales each name's deviations AROUND its own mean, so we can
               stress volatility without accidentally changing drift.
- drift      — additive daily return shock (uniform and/or per-ticker).
               Daily units: -0.002/day compounds to roughly -40% over a year,
               which is 2008-order-of-magnitude.

This is deliberately simple and transparent — the kind of stress test you can
explain to a risk committee in one sentence per scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from src.models.risk import (annualized_return, annualized_vol, hist_var,
                             max_drawdown, portfolio_returns)

BASELINE_NAME = "baseline"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    vol_mult: float = 1.0
    daily_drift: float = 0.0
    # per-ticker additive daily drift; unknown tickers are ignored so the
    # same scenario works for any configured portfolio.
    ticker_drift: Mapping[str, float] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario(
        "rate_shock",
        "Aggressive rate-hike cycle: long-duration assets (utilities, growth "
        "tech) repriced down, banks pick up margin; overall vol elevated.",
        vol_mult=1.3,
        ticker_drift={"NEE": -0.0015, "NVDA": -0.0010, "AAPL": -0.0008,
                      "MSFT": -0.0008, "JPM": +0.0005},
    ),
    Scenario(
        "market_crash_2008_style",
        "Systemic crash: everything sells off together (-0.2%/day drift, "
        "about -40% over a year) with volatility 2.5x normal.",
        vol_mult=2.5,
        daily_drift=-0.002,
    ),
    Scenario(
        "sector_shock_tech",
        "Tech-led selloff (AI unwind): heavy drift on NVDA/AAPL/MSFT, "
        "moderately elevated vol everywhere else.",
        vol_mult=1.4,
        ticker_drift={"NVDA": -0.0030, "AAPL": -0.0020, "MSFT": -0.0020},
    ),
]}


def apply_scenario(returns: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Return the shocked daily-returns panel."""
    mean = returns.mean()
    # scale deviations around each name's own mean -> pure vol shock ...
    shocked = mean + (returns - mean) * scenario.vol_mult
    # ... then layer drift shocks on top.
    shocked = shocked + scenario.daily_drift
    for ticker, drift in scenario.ticker_drift.items():
        if ticker in shocked.columns:
            shocked[ticker] = shocked[ticker] + drift
    return shocked


def portfolio_metrics(returns: pd.DataFrame) -> dict[str, float]:
    """The four numbers a stress comparison is judged on."""
    port = portfolio_returns(returns)
    return {
        "ann_return": float(annualized_return(port)),
        "ann_vol": float(annualized_vol(port)),
        "var_95": float(hist_var(port, 0.95)),
        "max_drawdown": float(max_drawdown(port)),
    }


def run_scenario(returns: pd.DataFrame, name: str) -> dict[str, float]:
    """Metrics for one named scenario (raises KeyError on unknown name)."""
    return portfolio_metrics(apply_scenario(returns, SCENARIOS[name]))


def compare(returns: pd.DataFrame,
            scenarios: Mapping[str, Scenario] | None = None) -> pd.DataFrame:
    """Baseline vs every scenario — the dashboard comparison table."""
    scenarios = scenarios or SCENARIOS
    rows = {BASELINE_NAME: portfolio_metrics(returns)}
    for name, scenario in scenarios.items():
        rows[name] = portfolio_metrics(apply_scenario(returns, scenario))
    return pd.DataFrame(rows).T.rename_axis("scenario")


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    print(compare(returns_wide()).round(4))
