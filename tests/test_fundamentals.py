"""Known-input checks for the fundamentals ratios and the EDGAR extraction.

No network: ratio math runs on a hand-built wide frame, and the loader's
parsing helpers run on synthetic companyfacts-shaped dicts.
"""
import numpy as np
import pandas as pd
import pytest

from src.ingest.edgar import _annual_points, _company_frame
from src.models.fundamentals import _safe_div, dupont, ratios


def wide(rows: dict) -> pd.DataFrame:
    """Build a (ticker x concept) frame from {ticker: {concept: value}}."""
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- ratio math

def test_ratios_match_hand_computed_values():
    w = wide({"ACME": {
        "current_assets": 200, "current_liabilities": 100, "inventory": 50,
        "revenue": 1000, "cogs": 600, "gross_profit": 400, "operating_income": 250,
        "net_income": 120, "assets": 800, "equity": 300, "liabilities": 500,
        "interest_expense": 25,
    }})
    r = ratios(w).loc["ACME"]
    assert r["current_ratio"] == pytest.approx(2.0)          # 200/100
    assert r["quick_ratio"] == pytest.approx(1.5)            # (200-50)/100
    assert r["debt_to_equity"] == pytest.approx(500 / 300)
    assert r["interest_coverage"] == pytest.approx(10.0)     # 250/25
    assert r["gross_margin"] == pytest.approx(0.4)           # 400/1000
    assert r["net_margin"] == pytest.approx(0.12)            # 120/1000
    assert r["roa"] == pytest.approx(0.15)                   # 120/800
    assert r["roe"] == pytest.approx(0.4)                    # 120/300
    assert r["asset_turnover"] == pytest.approx(1.25)        # 1000/800


def test_liabilities_derived_from_balance_sheet_identity():
    # No total-liabilities tag (the Walmart case): L must fall back to A - E.
    w = wide({"WMTish": {"assets": 300, "equity": 100, "net_income": 20, "revenue": 500}})
    assert ratios(w).loc["WMTish", "debt_to_equity"] == pytest.approx(2.0)  # (300-100)/100


def test_gross_profit_derived_when_untagged():
    w = wide({"X": {"revenue": 1000, "cogs": 700}})  # no gross_profit tag
    assert ratios(w).loc["X", "gross_margin"] == pytest.approx(0.3)  # (1000-700)/1000


def test_safe_div_zero_denominator_is_nan_not_inf():
    out = _safe_div(pd.Series([1.0]), pd.Series([0.0]))
    assert np.isnan(out.iloc[0])


# --------------------------------------------------------------- DuPont

def test_dupont_product_equals_direct_roe():
    w = wide({
        "AAPLish": {"revenue": 400, "net_income": 100, "assets": 350, "equity": 70},
        "WMTish":  {"revenue": 700, "net_income": 20, "assets": 280, "equity": 100},
    })
    d = dupont(w)
    # The decomposition must telescope: margin*turnover*multiplier == NI/Equity.
    pd.testing.assert_series_equal(d["roe"], d["roe_check"], check_names=False)
    assert d.loc["AAPLish", "roe"] == pytest.approx(100 / 70)


# --------------------------------------------------------------- EDGAR parsing

def _pt(end, val, *, form="10-K", start=None, filed="2025-01-01"):
    p = {"end": end, "val": val, "form": form, "filed": filed}
    if start:
        p["start"] = start
    return p


def test_annual_points_keeps_full_year_10k_only():
    fact = {"units": {"USD": [
        _pt("2024-12-31", 1000, start="2024-01-01"),          # annual 10-K -> keep
        _pt("2024-03-31", 250, start="2024-01-01"),           # quarter -> drop (90d)
        _pt("2023-12-31", 900, start="2023-01-01", form="10-Q"),  # not 10-K -> drop
    ]}}
    pts = _annual_points(fact)
    assert [p["val"] for p in pts] == [1000]


def test_annual_points_prefers_latest_filed_for_restated_period():
    fact = {"units": {"USD": [
        _pt("2023-12-31", 800, start="2023-01-01", filed="2024-02-01"),   # original
        _pt("2023-12-31", 820, start="2023-01-01", filed="2025-02-01"),   # restated
    ]}}
    pts = _annual_points(fact)
    assert [p["val"] for p in pts] == [820]  # most recently filed wins


def test_company_frame_picks_most_recent_tag_not_first():
    # The NVDA regression: an older candidate tag (listed first) has only stale
    # data; the newer tag the filer migrated to must win, or revenue pins to an
    # old year and can end up smaller than net income.
    facts = {"facts": {"us-gaap": {
        # first candidate for 'revenue', but stops at FY2022
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            _pt("2022-12-31", 27_000, start="2022-01-01"),
        ]}},
        # second candidate, current through FY2025
        "Revenues": {"units": {"USD": [
            _pt("2025-12-31", 130_000, start="2025-01-01"),
        ]}},
    }}}
    df = _company_frame(facts, "NVDAish")
    rev = df[df["concept"] == "revenue"]
    assert rev["fiscal_year"].max() == 2025
    assert rev.loc[rev["fiscal_year"].idxmax(), "value"] == 130_000
