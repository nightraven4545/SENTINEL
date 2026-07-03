"""Daily OHLCV ingestion via yfinance, with a local parquet cache.

Data is kept in LONG format (date, ticker, open, high, low, close, volume):
one row per ticker per day. That is the easiest shape to load into DuckDB
and can be pivoted wide on demand for the models.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# 10 names spread across sectors so the correlation / graph work later has
# real structure to find: tech (AAPL, MSFT, NVDA), banks (JPM), energy (XOM),
# pharma (JNJ), staples (PG, WMT), industrials (CAT), utilities (NEE).
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "NVDA", "CAT", "NEE", "WMT"]
DEFAULT_START = "2018-01-01"

# why 5: forward-fill only bridges short gaps (halts, bad prints). Anything
# longer probably means a delisting/data problem we should not paper over.
MAX_FFILL_DAYS = 5

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = DATA_DIR / "prices.parquet"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def get_tickers() -> list[str]:
    """Tickers from the SENTINEL_TICKERS env var, else the default portfolio."""
    raw = os.getenv("SENTINEL_TICKERS", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return tickers or list(DEFAULT_TICKERS)


def clean_prices(long: pd.DataFrame) -> pd.DataFrame:
    """Align tickers on a shared date grid and bridge short gaps.

    Forward-filling a missing close keeps the last traded price, which implies
    a 0% return for the gap day — the standard neutral assumption. Dropping
    the day instead would silently misalign tickers against each other.
    """
    dates = pd.Index(sorted(long["date"].unique()), name="date")
    tickers = pd.Index(sorted(long["ticker"].unique()), name="ticker")
    grid = pd.MultiIndex.from_product([dates, tickers])
    df = long.set_index(["date", "ticker"]).reindex(grid)
    df = df.groupby(level="ticker")[OHLCV_COLS].ffill(limit=MAX_FFILL_DAYS)
    # Rows still missing a close are pre-listing / unfixable — drop them.
    return df.dropna(subset=["close"]).reset_index()


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV, serving from the parquet cache when it covers
    the request (same-or-superset tickers, same-or-earlier start)."""
    tickers = tickers or get_tickers()

    if use_cache and CACHE_FILE.exists():
        cached = pd.read_parquet(CACHE_FILE)
        covers = set(tickers) <= set(cached["ticker"].unique()) and pd.Timestamp(
            cached["date"].min()
        ) <= pd.Timestamp(start) + pd.Timedelta(days=7)  # first trading day slack
        if covers:
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)

    # auto_adjust=True folds dividends/splits into prices, so close-to-close
    # pct changes are true total-ish returns — what every model downstream wants.
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    long = (
        raw.stack(level=0)  # (ticker, field) columns -> long rows
        .rename_axis(["date", "ticker"])
        .reset_index()
        .rename(columns=str.lower)
    )
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long = clean_prices(long[["date", "ticker", *OHLCV_COLS]])

    DATA_DIR.mkdir(exist_ok=True)
    long.to_parquet(CACHE_FILE, index=False)
    return long


if __name__ == "__main__":
    df = fetch_prices()
    print(f"{df['ticker'].nunique()} tickers, {len(df):,} rows, "
          f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(df.tail())
