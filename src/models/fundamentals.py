"""Financial-statement ratio analysis over SEC EDGAR fundamentals.

The accounting (CA) counterpart to risk.py: where that scores *market* risk from
prices, this scores *financial* health from the balance sheet, income statement
and cash-flow statement — the ratios a credit analyst or auditor reads first.

Input is the wide (ticker x concept) frame from
src.ingest.edgar.latest_fundamentals(); every function returns a ticker-indexed
DataFrame. Missing line items yield NaN rather than raising, because filers
legitimately differ (banks have no inventory; some skip a total-liabilities tag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.edgar import latest_fundamentals


def _col(wide: pd.DataFrame, name: str) -> pd.Series:
    """A concept column, or an all-NaN series if this universe never reports it."""
    if name in wide.columns:
        return wide[name]
    return pd.Series(np.nan, index=wide.index)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise ratio with divide-by-zero -> NaN (not inf), so downstream
    tables and charts never show a spurious ±infinity."""
    return (num / den.replace(0, np.nan))


def _fill_identities(wide: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct two commonly-untagged lines from accounting identities.

    total liabilities = assets − equity   (the balance-sheet identity A = L + E)
    gross profit       = revenue − COGS
    Filers often tag components but not these totals; deriving them keeps the
    ratio coverage complete without inventing data.
    """
    wide = wide.copy()
    liab = _col(wide, "liabilities")
    wide["liabilities"] = liab.fillna(_col(wide, "assets") - _col(wide, "equity"))
    gp = _col(wide, "gross_profit")
    wide["gross_profit"] = gp.fillna(_col(wide, "revenue") - _col(wide, "cogs"))
    return wide


def ratios(wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """The standard liquidity / solvency / profitability / efficiency battery.

    One row per ticker, latest fiscal year. Grouped the way an analyst reads a
    tear sheet: can it pay its bills, can it carry its debt, does it earn, and
    how hard do its assets work.
    """
    wide = latest_fundamentals() if wide is None else wide
    w = _fill_identities(wide)

    ca, cl = _col(w, "current_assets"), _col(w, "current_liabilities")
    rev, ni = _col(w, "revenue"), _col(w, "net_income")
    assets, equity = _col(w, "assets"), _col(w, "equity")

    out = pd.DataFrame(index=w.index)
    # Liquidity — short-term solvency
    out["current_ratio"] = _safe_div(ca, cl)
    out["quick_ratio"] = _safe_div(ca - _col(w, "inventory").fillna(0), cl)
    # Solvency / leverage — long-term structure
    out["debt_to_equity"] = _safe_div(_col(w, "liabilities"), equity)
    out["interest_coverage"] = _safe_div(_col(w, "operating_income"),
                                         _col(w, "interest_expense"))
    # Profitability — margins and returns
    out["gross_margin"] = _safe_div(_col(w, "gross_profit"), rev)
    out["operating_margin"] = _safe_div(_col(w, "operating_income"), rev)
    out["net_margin"] = _safe_div(ni, rev)
    out["roa"] = _safe_div(ni, assets)
    out["roe"] = _safe_div(ni, equity)
    # Efficiency
    out["asset_turnover"] = _safe_div(rev, assets)
    return out


def dupont(wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """3-step DuPont decomposition of ROE.

    ROE = net margin × asset turnover × equity multiplier
        = (NI/Sales) × (Sales/Assets) × (Assets/Equity)
    which telescopes back to NI/Equity. Splitting ROE this way shows *why* a
    company earns its return — fat margins, asset efficiency, or leverage — the
    single most-taught statement-analysis lens. `roe` (the product) and
    `roe_check` (NI/Equity directly) must match; any gap flags a data problem.
    """
    wide = latest_fundamentals() if wide is None else wide
    w = wide
    rev, ni = _col(w, "revenue"), _col(w, "net_income")
    assets, equity = _col(w, "assets"), _col(w, "equity")

    out = pd.DataFrame(index=w.index)
    out["net_margin"] = _safe_div(ni, rev)
    out["asset_turnover"] = _safe_div(rev, assets)
    out["equity_multiplier"] = _safe_div(assets, equity)
    out["roe"] = out["net_margin"] * out["asset_turnover"] * out["equity_multiplier"]
    out["roe_check"] = _safe_div(ni, equity)
    return out


if __name__ == "__main__":
    r = ratios()
    pd.set_option("display.width", 200)
    print("Ratios (latest FY):")
    print(r.round(2))
    print("\nDuPont ROE decomposition:")
    print(dupont().round(3))
