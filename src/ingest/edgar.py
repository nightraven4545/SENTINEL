"""SEC EDGAR fundamentals ingestion via the XBRL companyfacts API.

Where market.py gives Sentinel the *price* lens (CFA/market risk), this gives
the *accounting* lens (CA / financial-statement analysis): the actual income
statement, balance sheet and cash-flow line items a company reports in its
10-K, pulled straight from SEC's structured XBRL data — free, no key, just a
required User-Agent.

Kept in LONG format (ticker, concept, fiscal_year, end, value): one row per
reported annual figure. The ratio and forensic engines pivot what they need.

XBRL is messy: the same economic line item is tagged differently across filers
and over time (Revenues vs RevenueFromContractWithCustomer…). CONCEPTS maps each
canonical item to an ordered list of candidate us-gaap/dei tags; the first one a
filer actually reports wins.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingest.market import get_tickers

load_dotenv()

# SEC *requires* a descriptive User-Agent with contact info or it returns 403.
# Override with your own via .env; the default is the project owner's contact.
SEC_UA = os.getenv("SENTINEL_SEC_UA", "Sentinel research shreshtharawat4545@gmail.com")
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = DATA_DIR / "fundamentals.parquet"
# Same reasoning as prices_sample: SEC rate-limits/blocks shared cloud IPs, and
# a portfolio demo that 500s is worse than one on slightly stale filings.
SAMPLE_FILE = DATA_DIR / "samples" / "fundamentals_sample.parquet"

# Canonical line item -> ordered candidate tags (taxonomy, tag). Superset for
# both the ratio engine (this session) and the forensic scores (next session),
# so we fetch once. "dei" is SEC's entity-info taxonomy (share counts live there).
CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "revenue": [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "Revenues"), ("us-gaap", "SalesRevenueNet")],
    "cogs": [("us-gaap", "CostOfGoodsAndServicesSold"), ("us-gaap", "CostOfRevenue"),
             ("us-gaap", "CostOfGoodsSold")],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "assets": [("us-gaap", "Assets")],
    "current_assets": [("us-gaap", "AssetsCurrent")],
    "liabilities": [("us-gaap", "Liabilities")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "retained_earnings": [("us-gaap", "RetainedEarningsAccumulatedDeficit")],
    "cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
             ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")],
    "inventory": [("us-gaap", "InventoryNet")],
    "receivables": [("us-gaap", "AccountsReceivableNetCurrent"),
                    ("us-gaap", "ReceivablesNetCurrent")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "ppe_net": [("us-gaap", "PropertyPlantAndEquipmentNet")],
    "depreciation": [("us-gaap", "DepreciationDepletionAndAmortization"),
                     ("us-gaap", "DepreciationAmortizationAndAccretionNet"),
                     ("us-gaap", "Depreciation")],
    "sga": [("us-gaap", "SellingGeneralAndAdministrativeExpense"),
            ("us-gaap", "GeneralAndAdministrativeExpense")],
    "interest_expense": [("us-gaap", "InterestExpense"), ("us-gaap", "InterestExpenseDebt")],
    "income_tax": [("us-gaap", "IncomeTaxExpenseBenefit")],
    "long_term_debt": [("us-gaap", "LongTermDebtNoncurrent"), ("us-gaap", "LongTermDebt")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding"),
               ("us-gaap", "CommonStockSharesOutstanding")],
}

# Annual figures only: a 10-K's flow items (revenue, income) span a full year;
# quarterly rows also appear in the file, so require the period to be ~1 year.
_YEAR_MIN_DAYS, _YEAR_MAX_DAYS = 350, 380


def _get(url: str, _tries: int = 3) -> dict:
    """GET + parse JSON with the required SEC User-Agent and a polite retry.

    SEC asks callers to stay under ~10 req/s; a short sleep between the handful
    of requests we make keeps us well-behaved and dodges transient throttling.
    """
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    for attempt in range(_tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == _tries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    return {}  # unreachable; keeps type checkers happy


def cik_map() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, from SEC's master ticker file."""
    data = _get(_TICKERS_URL)
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}


def _annual_points(fact: dict) -> list[dict]:
    """Extract clean annual 10-K rows from one companyfacts tag entry.

    Keeps only 10-K figures; for flow items (those with a start date) requires a
    ~1-year period so quarterly rows in the filing are excluded. When the same
    period end appears more than once (original vs amended/restated), the most
    recently *filed* value wins.
    """
    units = fact.get("units", {})
    # Values are almost always USD; share counts come in "shares". Take whichever
    # unit this tag actually uses (there is normally exactly one relevant unit).
    unit_key = next((u for u in ("USD", "shares") if u in units), None)
    if unit_key is None:
        return []

    by_end: dict[str, dict] = {}
    for p in units[unit_key]:
        if not str(p.get("form", "")).startswith("10-K"):
            continue
        if "start" in p:  # flow item — must span a full fiscal year
            days = (pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])).days
            if not (_YEAR_MIN_DAYS <= days <= _YEAR_MAX_DAYS):
                continue
        prev = by_end.get(p["end"])
        if prev is None or p.get("filed", "") > prev.get("filed", ""):
            by_end[p["end"]] = p
    return list(by_end.values())


def _company_frame(facts: dict, ticker: str) -> pd.DataFrame:
    """Long frame of every canonical concept for one company across all years."""
    rows: list[dict] = []
    taxo = facts.get("facts", {})
    for concept, candidates in CONCEPTS.items():
        # Pick the candidate tag with the most RECENT coverage, not merely the
        # first non-empty one: filers migrate tags over time (e.g. NVDA moved off
        # RevenueFromContractWithCustomer…), and a stale tag would silently pin
        # the value to an old year. Recency, then point count, breaks ties.
        best: list[dict] = []
        best_key = (pd.Timestamp.min, 0)
        for taxonomy, tag in candidates:
            fact = taxo.get(taxonomy, {}).get(tag)
            points = _annual_points(fact) if fact else []
            if not points:
                continue
            key = (max(pd.Timestamp(p["end"]) for p in points), len(points))
            if key > best_key:
                best_key, best = key, points
        for p in best:
            end = pd.Timestamp(p["end"])
            rows.append({"ticker": ticker, "concept": concept,
                         "fiscal_year": end.year, "end": end,
                         "value": float(p["val"])})
    return pd.DataFrame(rows)


def fetch_fundamentals(
    tickers: list[str] | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Annual fundamentals in long format for the configured universe.

    Resilience ladder mirrors market.fetch_prices: local parquet cache ->
    SEC network -> committed sample snapshot (cloud hosts get blocked by SEC).
    """
    tickers = tickers or get_tickers()

    if use_cache and CACHE_FILE.exists():
        cached = pd.read_parquet(CACHE_FILE)
        if set(tickers) <= set(cached["ticker"].unique()):
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)

    try:
        cik = cik_map()
        frames = []
        for t in tickers:
            if t not in cik:
                continue
            facts = _get(_FACTS_URL.format(cik=cik[t]))
            frames.append(_company_frame(facts, t))
            time.sleep(0.2)  # stay comfortably under SEC's rate limit
        long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception:
        long = pd.DataFrame()

    if long.empty:
        if SAMPLE_FILE.exists():  # blocked/offline -> bundled snapshot
            sample = pd.read_parquet(SAMPLE_FILE)
            return sample[sample["ticker"].isin(tickers)].reset_index(drop=True)
        raise RuntimeError(f"SEC returned no fundamentals for {tickers}")

    DATA_DIR.mkdir(exist_ok=True)
    long.to_parquet(CACHE_FILE, index=False)
    return long


def latest_fundamentals(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Most recent fiscal year per ticker, pivoted wide (ticker x concept).

    The ratio engine works cross-sectionally on one year; the long frame keeps
    full history for the forensic scores (which need year-over-year deltas).
    """
    long = fetch_fundamentals() if long is None else long
    # Each concept's own most recent annual value: the dei share count is often
    # a year ahead of the 10-K financials, so a single global "latest year" per
    # ticker would drop whichever items lag it. Core statement lines are filed
    # together, so they still align to the same fiscal year in practice.
    latest = long.loc[long.groupby(["ticker", "concept"])["end"].idxmax()]
    wide = latest.pivot_table(index="ticker", columns="concept", values="value",
                              aggfunc="last")
    return wide


if __name__ == "__main__":
    df = fetch_fundamentals()
    print(f"{df['ticker'].nunique()} companies, {len(df):,} annual facts, "
          f"FY {df['fiscal_year'].min()}–{df['fiscal_year'].max()}")
    wide = latest_fundamentals(df)
    print(wide[["revenue", "net_income", "assets", "equity"]].round(0))
