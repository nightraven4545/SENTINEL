"""Tests for the ingest cleaner and the DuckDB warehouse (no network)."""
import pandas as pd
import pytest

from src.ingest.market import clean_prices, get_tickers
from src.warehouse.duck import connect, load_prices, query


def make_long(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low"):
        df[col] = df["close"]
    df["volume"] = 1000.0
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def test_clean_prices_fills_short_gaps():
    # B is missing on day 2 -> should be forward-filled from day 1.
    long = make_long([
        ("2024-01-01", "A", 10.0), ("2024-01-02", "A", 11.0), ("2024-01-03", "A", 12.0),
        ("2024-01-01", "B", 20.0), ("2024-01-03", "B", 22.0),
    ])
    cleaned = clean_prices(long)
    b_day2 = cleaned[(cleaned["ticker"] == "B") & (cleaned["date"] == "2024-01-02")]
    assert len(b_day2) == 1
    assert b_day2["close"].iloc[0] == pytest.approx(20.0)


def test_clean_prices_drops_pre_listing_rows():
    # B only exists from day 3 -> days 1-2 must not be invented backwards.
    long = make_long([
        ("2024-01-01", "A", 10.0), ("2024-01-02", "A", 11.0), ("2024-01-03", "A", 12.0),
        ("2024-01-03", "B", 22.0),
    ])
    cleaned = clean_prices(long)
    assert len(cleaned[cleaned["ticker"] == "B"]) == 1


def test_get_tickers_env_override(monkeypatch):
    monkeypatch.setenv("SENTINEL_TICKERS", "spy, qqq")
    assert get_tickers() == ["SPY", "QQQ"]


def test_warehouse_load_and_derived_returns(tmp_path):
    db = tmp_path / "test.duckdb"
    long = make_long([
        ("2024-01-01", "A", 100.0), ("2024-01-02", "A", 110.0), ("2024-01-03", "A", 99.0),
    ])
    con = connect(db)
    load_prices(con, long)
    con.close()

    rets = query("SELECT * FROM returns ORDER BY date", db)
    # Day 1 has no prior close -> only 2 return rows; check the math.
    assert len(rets) == 2
    assert rets["ret"].iloc[0] == pytest.approx(0.10)
    assert rets["ret"].iloc[1] == pytest.approx(99 / 110 - 1)


def test_warehouse_load_is_idempotent(tmp_path):
    db = tmp_path / "test.duckdb"
    long = make_long([("2024-01-01", "A", 100.0), ("2024-01-02", "A", 101.0)])
    con = connect(db)
    load_prices(con, long)
    load_prices(con, long)  # replace-load: no duplicate-key explosion
    con.close()
    assert len(query("SELECT * FROM prices", db)) == 2
