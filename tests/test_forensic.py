"""Hand-computed checks for the forensic scores.

No network: every score runs on a synthetic long frame with values chosen so the
expected output can be worked out by hand (the Beneish M-score in particular is
verified index-by-index).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.forensic import (
    BENFORD_EXPECTED,
    _leading_digit,
    accruals_ratio,
    altman_z,
    beneish_m,
    benford_mad,
    piotroski_f,
)


def long_frame(rows) -> pd.DataFrame:
    """rows: list of (ticker, concept, end_str, value) -> long fundamentals frame."""
    df = pd.DataFrame(rows, columns=["ticker", "concept", "end", "value"])
    df["end"] = pd.to_datetime(df["end"])
    df["fiscal_year"] = df["end"].dt.year
    return df


def one_year(ticker, end, **concepts):
    return [(ticker, c, end, v) for c, v in concepts.items()]


# ------------------------------------------------------------------ Altman Z

def test_altman_z_and_zone_hand_computed():
    lf = long_frame(one_year(
        "ACME", "2024-12-31", assets=1000, current_assets=400,
        current_liabilities=200, retained_earnings=300, operating_income=150,
        liabilities=500, revenue=900))
    mcap = pd.Series({"ACME": 800.0})
    z = altman_z(lf, market_cap=mcap).loc["ACME"]
    # X1=.2 X2=.3 X3=.15 X4=1.6 X5=.9
    # Z = 1.2*.2 + 1.4*.3 + 3.3*.15 + .6*1.6 + 1.0*.9 = 3.015
    assert z["z_score"] == pytest.approx(3.015)
    assert z["zone"] == "safe"  # > 2.99


def test_altman_ebit_falls_back_to_pretax_plus_interest():
    # No operating_income tag: EBIT = net_income + tax + interest = 150.
    lf = long_frame(one_year(
        "NOOP", "2024-12-31", assets=1000, current_assets=400,
        current_liabilities=200, retained_earnings=300, net_income=100,
        income_tax=30, interest_expense=20, liabilities=500, revenue=900))
    z = altman_z(lf, market_cap=pd.Series({"NOOP": 800.0})).loc["NOOP"]
    assert z["X3_ebit"] == pytest.approx(0.15)      # 150/1000
    assert z["z_score"] == pytest.approx(3.015)


# ------------------------------------------------------------------ Piotroski F

def test_piotroski_perfect_nine():
    rows = one_year("GOOD", "2023-12-31", net_income=50, assets=1000, cfo=40,
                    long_term_debt=200, current_assets=300, current_liabilities=200,
                    shares=100, gross_profit=400, revenue=900)
    rows += one_year("GOOD", "2024-12-31", net_income=100, assets=1000, cfo=120,
                     long_term_debt=100, current_assets=400, current_liabilities=200,
                     shares=90, gross_profit=500, revenue=1000)
    f = piotroski_f(long_frame(rows)).loc["GOOD"]
    assert f["f_score"] == 9.0  # every signal improves


def test_piotroski_weak_company_scores_low():
    # Deteriorating on every axis + losses -> near zero.
    rows = one_year("BAD", "2023-12-31", net_income=50, assets=1000, cfo=60,
                    long_term_debt=100, current_assets=400, current_liabilities=200,
                    shares=100, gross_profit=500, revenue=1000)
    rows += one_year("BAD", "2024-12-31", net_income=-20, assets=1000, cfo=-10,
                     long_term_debt=200, current_assets=300, current_liabilities=200,
                     shares=120, gross_profit=400, revenue=900)
    f = piotroski_f(long_frame(rows)).loc["BAD"]
    assert f["f_score"] <= 1.0


# ------------------------------------------------------------------ Beneish M

def test_beneish_m_index_by_index():
    rows = one_year("MANIP", "2023-12-31", revenue=1000, receivables=100,
                    gross_profit=400, assets=2000, current_assets=800, ppe_net=500,
                    depreciation=50, sga=100, long_term_debt=300,
                    current_liabilities=200, net_income=70, cfo=80)
    rows += one_year("MANIP", "2024-12-31", revenue=1200, receivables=150,
                     gross_profit=500, assets=2500, current_assets=900, ppe_net=600,
                     depreciation=60, sga=130, long_term_debt=400,
                     current_liabilities=250, net_income=150, cfo=120)
    m = beneish_m(long_frame(rows)).loc["MANIP"]
    assert m["DSRI"] == pytest.approx(1.25)        # (150/1200)/(100/1000)
    assert m["GMI"] == pytest.approx(0.96)         # (400/1000)/(500/1200)
    assert m["AQI"] == pytest.approx(1.142857, abs=1e-5)
    assert m["SGI"] == pytest.approx(1.2)          # 1200/1000
    assert m["DEPI"] == pytest.approx(1.0)         # dep rates equal
    assert m["SGAI"] == pytest.approx(1.083333, abs=1e-5)
    assert m["LVGI"] == pytest.approx(1.04)        # .26/.25
    assert m["TATA"] == pytest.approx(0.012)       # (150-120)/2500
    # M assembled from the coefficients -> ~ -2.006, below the -1.78 flag line.
    assert m["m_score"] == pytest.approx(-2.006, abs=5e-3)
    assert m["manipulation_flag"] is False or m["manipulation_flag"] == False  # noqa: E712


# ------------------------------------------------------------------ Accruals

def test_accruals_uses_average_assets():
    rows = one_year("A", "2023-12-31", assets=800)
    rows += one_year("A", "2024-12-31", assets=1000, net_income=100, cfo=120)
    # (100-120) / mean(1000,800) = -20/900
    assert accruals_ratio(long_frame(rows)).loc["A"] == pytest.approx(-20 / 900)


# ------------------------------------------------------------------ fiscal alignment

def test_cover_page_shares_align_to_financial_year():
    # Dec-end filer: balance sheet 2024-12-31, cover-page shares 2025-01-31 must
    # cluster into the SAME fiscal year, not spawn a phantom 2025 row.
    rows = [("DEC", "assets", "2024-12-31", 1000),
            ("DEC", "net_income", "2024-12-31", 100),
            ("DEC", "shares", "2025-01-31", 42),          # cover date, next Jan
            ("DEC", "assets", "2023-12-31", 900)]
    f = piotroski_f(long_frame(rows))
    # If alignment failed, the latest row would be the shares-only phantom and
    # ROA (needs net_income & assets together) would be NaN -> roa_pos == 0.
    assert f.loc["DEC", "roa_pos"] == 1.0


# ------------------------------------------------------------------ Benford

def test_leading_digit_extraction():
    d = _leading_digit(pd.Series([1.0, 23.0, 0.045, 900.0, -7.0, 0.0]))
    assert sorted(d.tolist()) == [1, 2, 4, 7, 9]  # zero dropped


def test_benford_conformity_on_benford_distributed_digits():
    # Build values whose leading digits follow Benford exactly (number d -> digit d).
    rows = []
    for d in range(1, 10):
        for _ in range(int(round(BENFORD_EXPECTED[d] * 5000))):
            rows.append(("U", "revenue", "2024-12-31", float(d)))
    res = benford_mad(long_frame(rows))
    assert res["verdict"] == "close conformity"
    assert res["mad"] < 0.006


def test_benford_flags_single_digit_dominance():
    rows = [("U", "revenue", "2024-12-31", 100.0) for _ in range(500)]  # all lead 1
    res = benford_mad(long_frame(rows))
    assert res["verdict"] == "nonconformity"
