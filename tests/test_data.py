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


def test_fetch_prices_falls_back_to_bundled_sample(tmp_path, monkeypatch):
    import src.ingest.market as market

    sample = make_long([("2024-01-01", "A", 10.0), ("2024-01-01", "B", 20.0)])
    sample_file = tmp_path / "prices_sample.parquet"
    sample.to_parquet(sample_file, index=False)
    monkeypatch.setattr(market, "SAMPLE_FILE", sample_file)
    monkeypatch.setattr(market, "CACHE_FILE", tmp_path / "no_cache.parquet")
    monkeypatch.setattr(market.yf, "download",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("rate limited")))

    got = market.fetch_prices(tickers=["A"], use_cache=False)
    assert list(got["ticker"].unique()) == ["A"]


def test_ensure_loaded_bootstraps_and_recovers_half_written_db(tmp_path, monkeypatch):
    import src.ingest.market as market
    from src.warehouse.duck import connect, ensure_loaded, query

    long = make_long([("2024-01-01", "A", 100.0), ("2024-01-02", "A", 110.0)])
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return long

    monkeypatch.setattr(market, "fetch_prices", fake_fetch)
    db = tmp_path / "w.duckdb"

    # simulate the cloud failure mode: schema-only file left behind
    connect(db).close()
    ensure_loaded(db)
    assert query("SELECT count(*) AS n FROM returns", db)["n"].iat[0] == 1
    assert calls["n"] == 1

    ensure_loaded(db)  # already populated -> no refetch
    assert calls["n"] == 1


def test_warehouse_load_is_idempotent(tmp_path):
    db = tmp_path / "test.duckdb"
    long = make_long([("2024-01-01", "A", 100.0), ("2024-01-02", "A", 101.0)])
    con = connect(db)
    load_prices(con, long)
    load_prices(con, long)  # replace-load: no duplicate-key explosion
    con.close()
    assert len(query("SELECT * FROM prices", db)) == 2
