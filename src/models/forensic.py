"""Forensic-accounting scores over multi-year EDGAR fundamentals.

This is where Sentinel's "anomaly engine" thesis reaches the financial
statements: instead of flagging odd *price* days, these screens flag odd
*accounting* — the models auditors, short-sellers and credit desks use to spot
distress and earnings manipulation before the market does.

Five classics:
  • Altman Z-score   — bankruptcy risk (distance from insolvency)
  • Piotroski F-score — 9-point fundamental-quality checklist
  • Beneish M-score   — probability the books are being manipulated
  • Accruals ratio    — how much of earnings is *not* backed by cash (Sloan)
  • Benford's Law     — do the reported digits look naturally occurring?

Input is the LONG frame from src.ingest.edgar.fetch_fundamentals()
(ticker, concept, fiscal_year, end, value). Scores needing year-over-year change
(Beneish, Piotroski) use the two most recent fiscal years; a name with only one
year of data yields NaN rather than a wrong number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.edgar import fetch_fundamentals
from src.models.fundamentals import _col, _safe_div

# Benford's Law: the frequency of leading digit d in naturally-occurring data is
# log10(1 + 1/d). Manipulated figures tend to deviate — the basis of digit
# analysis in forensic auditing (Nigrini).
BENFORD_EXPECTED = pd.Series(
    {d: np.log10(1 + 1 / d) for d in range(1, 10)}, name="expected")

# Nigrini's mean-absolute-deviation conformity thresholds for first digits.
_MAD_CLOSE, _MAD_ACCEPTABLE, _MAD_MARGINAL = 0.006, 0.012, 0.015

# Altman (1968) distress zones for the original public-manufacturer model.
_Z_SAFE, _Z_DISTRESS = 2.99, 1.81
# Beneish (1999): above this, the 8-variable model classifies a likely manipulator.
_M_THRESHOLD = -1.78


# ------------------------------------------------------------------ data shaping

def _fiscal_panel(long: pd.DataFrame) -> pd.DataFrame:
    """(ticker, fiscal_year) × concept panel with reporting dates clustered.

    Raw fiscal_year = end.year splits a fiscal period in two for December-end
    filers: the balance sheet ends Dec-31 but the 10-K cover-page share count is
    dated late-January, landing in the next calendar year and creating a phantom,
    near-empty "latest" row. Clustering each filer's report dates (a >150-day gap
    starts a new fiscal period) keeps a period's statements and its share count
    together; the period is labelled by its earliest end — the fiscal-year end,
    which always precedes the cover date.
    """
    long = long.sort_values(["ticker", "end"]).copy()
    gap = long.groupby("ticker")["end"].diff().dt.days
    long["_period"] = (gap.isna() | (gap > 150)).groupby(long["ticker"]).cumsum()
    long["_fy"] = long.groupby(["ticker", "_period"])["end"].transform("min").dt.year
    return long.pivot_table(index=["ticker", "_fy"], columns="concept",
                            values="value", aggfunc="last").sort_index()


def _curr_prev(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two ticker×concept frames: the latest fiscal year and the one before it.

    Aligning both to ticker-indexed wide frames lets every year-over-year index
    below be a clean elementwise ratio. Single-year names get an all-NaN prev.
    """
    panel = _fiscal_panel(long)
    curr, prev = {}, {}
    for tkr, sub in panel.groupby(level="ticker"):
        sub = sub.droplevel("ticker")
        curr[tkr] = sub.iloc[-1]
        prev[tkr] = sub.iloc[-2] if len(sub) >= 2 else pd.Series(np.nan, index=sub.columns)
    return pd.DataFrame(curr).T, pd.DataFrame(prev).T


def _total_liabilities(w: pd.DataFrame) -> pd.Series:
    """Total liabilities, reconstructed as assets − equity when untagged
    (balance-sheet identity A = L + E; some filers skip the total line)."""
    return _col(w, "liabilities").fillna(_col(w, "assets") - _col(w, "equity"))


# ------------------------------------------------------------------ Altman Z

def altman_z(long: pd.DataFrame | None = None,
             market_cap: pd.Series | None = None) -> pd.DataFrame:
    """Altman (1968) Z-score for public manufacturers.

    Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets                       (EBIT ≈ operating income)
      X4 = market value of equity / total liabilities
      X5 = sales / total assets
    Zones: Z > 2.99 safe · 1.81–2.99 grey · < 1.81 distress.

    X4 wants MARKET value of equity (shares × price). If `market_cap` is not
    supplied it falls back to *book* equity — a documented approximation that
    understates X4 for names trading above book. The model is calibrated on
    manufacturers, so it is not meaningful for banks/financials (flagged NaN-ish
    results should be read with that caveat).
    """
    curr, _ = _curr_prev(fetch_fundamentals() if long is None else long)
    assets = _col(curr, "assets")
    liabilities = _total_liabilities(curr)
    mve = market_cap if market_cap is not None else _col(curr, "equity")
    mve = mve.reindex(curr.index)

    # EBIT: operating income when tagged, else reconstructed as
    # pre-tax income + interest = net income + tax + interest expense (some
    # filers, e.g. JNJ/XOM, don't tag OperatingIncomeLoss).
    ebit = _col(curr, "operating_income").fillna(
        _col(curr, "net_income") + _col(curr, "income_tax")
        + _col(curr, "interest_expense").fillna(0))

    x1 = _safe_div(_col(curr, "current_assets") - _col(curr, "current_liabilities"), assets)
    x2 = _safe_div(_col(curr, "retained_earnings"), assets)
    x3 = _safe_div(ebit, assets)
    x4 = _safe_div(mve, liabilities)
    x5 = _safe_div(_col(curr, "revenue"), assets)

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    out = pd.DataFrame({"X1_wc": x1, "X2_re": x2, "X3_ebit": x3,
                        "X4_mve": x4, "X5_sales": x5, "z_score": z})
    out["zone"] = np.where(z > _Z_SAFE, "safe",
                           np.where(z < _Z_DISTRESS, "distress", "grey"))
    out.loc[z.isna(), "zone"] = None
    return out


# ------------------------------------------------------------------ Piotroski F

def piotroski_f(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Piotroski (2000) F-score: nine binary fundamental-health signals (0–9).

    Profitability: ROA>0, CFO>0, ΔROA>0, accrual (CFO>Net Income).
    Leverage/liquidity: lower long-term leverage, higher current ratio, no new shares.
    Efficiency: higher gross margin, higher asset turnover.
    8–9 = strong fundamentals; 0–2 = weak. Each signal is +1 when it *improves*.
    """
    curr, prev = _curr_prev(fetch_fundamentals() if long is None else long)

    def roa(w):  # return on assets
        return _safe_div(_col(w, "net_income"), _col(w, "assets"))

    def lever(w):  # long-term debt / assets
        return _safe_div(_col(w, "long_term_debt"), _col(w, "assets"))

    def curr_ratio(w):
        return _safe_div(_col(w, "current_assets"), _col(w, "current_liabilities"))

    def gmargin(w):
        gp = _col(w, "gross_profit").fillna(_col(w, "revenue") - _col(w, "cogs"))
        return _safe_div(gp, _col(w, "revenue"))

    def turnover(w):
        return _safe_div(_col(w, "revenue"), _col(w, "assets"))

    cfo = _col(curr, "cfo")
    signals = pd.DataFrame(index=curr.index)
    signals["roa_pos"] = (roa(curr) > 0).astype(float)
    signals["cfo_pos"] = (cfo > 0).astype(float)
    signals["droa_pos"] = (roa(curr) > roa(prev)).astype(float)
    # accrual: cash earnings exceed accounting earnings -> higher quality
    signals["accrual"] = (cfo > _col(curr, "net_income")).astype(float)
    signals["dlever_neg"] = (lever(curr) < lever(prev)).astype(float)
    signals["dliquid_pos"] = (curr_ratio(curr) > curr_ratio(prev)).astype(float)
    # no new shares issued (dilution) — allow tiny rounding wiggle
    signals["no_dilution"] = (_col(curr, "shares") <= _col(prev, "shares") * 1.001).astype(float)
    signals["dmargin_pos"] = (gmargin(curr) > gmargin(prev)).astype(float)
    signals["dturn_pos"] = (turnover(curr) > turnover(prev)).astype(float)

    # A signal that can't be evaluated (missing prior year / inputs) scores 0,
    # matching Piotroski's convention that an unmet/unknown criterion earns no point.
    signals["f_score"] = signals.sum(axis=1)
    return signals


# ------------------------------------------------------------------ Beneish M

def beneish_m(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Beneish (1999) M-score: 8-variable earnings-manipulation probability.

    M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI
            − 0.172·SGAI + 4.679·TATA − 0.327·LVGI
    M > −1.78 flags a likely manipulator. (Beneish's model famously scored Enron
    as a manipulator well before its collapse.) Each index compares this year to
    last; note GMI and DEPI are prior÷current (a *rising* number = deterioration).
    """
    curr, prev = _curr_prev(fetch_fundamentals() if long is None else long)

    def gmargin(w):
        gp = _col(w, "gross_profit").fillna(_col(w, "revenue") - _col(w, "cogs"))
        return _safe_div(gp, _col(w, "revenue"))

    def asset_quality(w):  # 1 − (current assets + net PP&E)/assets = "soft" assets
        hard = _col(w, "current_assets") + _col(w, "ppe_net")
        return 1 - _safe_div(hard, _col(w, "assets"))

    def dep_rate(w):  # depreciation / (depreciation + net PP&E)
        dep = _col(w, "depreciation")
        return _safe_div(dep, dep + _col(w, "ppe_net"))

    def leverage(w):  # (LT debt + current liabilities) / assets
        return _safe_div(_col(w, "long_term_debt") + _col(w, "current_liabilities"),
                         _col(w, "assets"))

    rev_c, rev_p = _col(curr, "revenue"), _col(prev, "revenue")
    dsri = _safe_div(_safe_div(_col(curr, "receivables"), rev_c),
                     _safe_div(_col(prev, "receivables"), rev_p))
    gmi = _safe_div(gmargin(prev), gmargin(curr))          # prior ÷ current
    aqi = _safe_div(asset_quality(curr), asset_quality(prev))
    sgi = _safe_div(rev_c, rev_p)
    depi = _safe_div(dep_rate(prev), dep_rate(curr))       # prior ÷ current
    sgai = _safe_div(_safe_div(_col(curr, "sga"), rev_c),
                     _safe_div(_col(prev, "sga"), rev_p))
    lvgi = _safe_div(leverage(curr), leverage(prev))
    # total accruals = (net income − cash from operations) / total assets
    tata = _safe_div(_col(curr, "net_income") - _col(curr, "cfo"), _col(curr, "assets"))

    m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    out = pd.DataFrame({"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
                        "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata,
                        "m_score": m})
    # object dtype so the flag can be True / False / None (pandas 3 won't let a
    # bool column hold NaN for names lacking a prior year).
    out["manipulation_flag"] = pd.Series(
        np.where(m.isna(), None, m > _M_THRESHOLD), index=m.index, dtype=object)
    return out


# ------------------------------------------------------------------ Accruals

def accruals_ratio(long: pd.DataFrame | None = None) -> pd.Series:
    """Sloan (1996) balance-of-earnings accruals: (Net Income − CFO) / avg assets.

    High positive accruals mean earnings are running ahead of the cash actually
    collected — the classic low-earnings-quality / future-underperformance signal.
    Averages current and prior total assets when both are available.
    """
    curr, prev = _curr_prev(fetch_fundamentals() if long is None else long)
    avg_assets = pd.concat([_col(curr, "assets"), _col(prev, "assets")], axis=1).mean(axis=1)
    return _safe_div(_col(curr, "net_income") - _col(curr, "cfo"), avg_assets).rename("accruals_ratio")


# ------------------------------------------------------------------ Benford

def _leading_digit(values: pd.Series) -> pd.Series:
    """First significant digit (1–9) of each nonzero magnitude."""
    v = values.astype(float).abs()
    v = v[v > 0]
    return (v / 10 ** np.floor(np.log10(v))).astype(int)


def benford_distribution(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observed vs expected first-digit frequency over all reported figures."""
    long = fetch_fundamentals() if long is None else long
    digits = _leading_digit(long["value"])
    observed = digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0.0)
    observed.index.name = "digit"
    return pd.DataFrame({"observed": observed, "expected": BENFORD_EXPECTED})


def _mad_verdict(mad: float) -> str:
    if mad < _MAD_CLOSE:
        return "close conformity"
    if mad < _MAD_ACCEPTABLE:
        return "acceptable conformity"
    if mad < _MAD_MARGINAL:
        return "marginal"
    return "nonconformity"


def benford_mad(long: pd.DataFrame | None = None) -> dict:
    """Mean absolute deviation of first digits from Benford + Nigrini verdict."""
    dist = benford_distribution(long)
    mad = float((dist["observed"] - dist["expected"]).abs().mean())
    n = int(_leading_digit((fetch_fundamentals() if long is None else long)["value"]).shape[0])
    return {"mad": mad, "verdict": _mad_verdict(mad), "n": n}


# ------------------------------------------------------------------ summary

def forensic_summary(long: pd.DataFrame | None = None,
                     market_cap: pd.Series | None = None) -> pd.DataFrame:
    """One row per ticker: the headline number from each forensic screen."""
    long = fetch_fundamentals() if long is None else long
    z = altman_z(long, market_cap)
    f = piotroski_f(long)
    m = beneish_m(long)
    return pd.DataFrame({
        "altman_z": z["z_score"],
        "z_zone": z["zone"],
        "piotroski_f": f["f_score"],
        "beneish_m": m["m_score"],
        "m_flag": m["manipulation_flag"],
        "accruals": accruals_ratio(long),
    })


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("Forensic summary:")
    print(forensic_summary().round(2))
    print("\nBenford (universe):", benford_mad())
