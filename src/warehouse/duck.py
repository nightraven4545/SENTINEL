"""DuckDB warehouse: schema, load, and a query() helper.

DuckDB is a local analytical database — same mental model as a warehouse
(tables + SQL) without any server to run. The file lives in data/ and is
gitignored.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_FILE = DATA_DIR / "sentinel.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date   DATE,
    ticker VARCHAR,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume BIGINT,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS returns (
    date   DATE,
    ticker VARCHAR,
    ret    DOUBLE,
    PRIMARY KEY (date, ticker)
);
"""


def connect(db_path: Path | str = DB_FILE) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse and ensure the schema exists."""
    Path(db_path).parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def load_prices(con: duckdb.DuckDBPyConnection, prices: pd.DataFrame) -> None:
    """Replace-load the prices table and derive daily returns in SQL.

    Returns are close-to-close pct changes per ticker; computing them in the
    warehouse (window function) keeps a single source of truth for every
    consumer instead of each model re-deriving them slightly differently.
    """
    con.register("prices_df", prices)
    con.execute("DELETE FROM prices")
    con.execute("""
        INSERT INTO prices
        SELECT date, ticker, open, high, low, close, CAST(volume AS BIGINT)
        FROM prices_df
    """)
    con.execute("DELETE FROM returns")
    con.execute("""
        INSERT INTO returns
        SELECT date, ticker,
               close / lag(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
        FROM prices
        QUALIFY ret IS NOT NULL
    """)
    con.unregister("prices_df")


def query(sql: str, db_path: Path | str = DB_FILE) -> pd.DataFrame:
    """Run a read-only SQL query against the warehouse, returning a DataFrame."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(sql).df()


def ensure_loaded(db_path: Path | str = DB_FILE) -> None:
    """Bootstrap the warehouse if it's missing, empty, or half-written.

    Order matters: fetch the data BEFORE opening the write connection, so a
    failed fetch (cloud hosts get rate-limited by Yahoo) can never leave
    behind an empty, write-locked database file — which is exactly the
    failure mode that bricks a fresh Streamlit Cloud deploy.
    """
    db_path = Path(db_path)
    if db_path.exists():
        try:
            if query("SELECT count(*) FROM returns", db_path).iat[0, 0] > 0:
                return
        except Exception:
            pass  # unreadable / schema-only file: rebuild from scratch
        db_path.unlink()

    from src.ingest.market import fetch_prices
    prices = fetch_prices()
    con = connect(db_path)
    try:
        load_prices(con, prices)
    finally:
        con.close()


def returns_wide(db_path: Path | str = DB_FILE) -> pd.DataFrame:
    """Daily returns pivoted wide (date index, one column per ticker) —
    the shape every model in src/models consumes."""
    long = query("SELECT date, ticker, ret FROM returns ORDER BY date", db_path)
    wide = long.pivot(index="date", columns="ticker", values="ret")
    wide.index = pd.to_datetime(wide.index)
    return wide


if __name__ == "__main__":
    from src.ingest.market import fetch_prices

    con = connect()
    load_prices(con, fetch_prices())
    con.close()
    print(query("SELECT ticker, count(*) AS days, min(date) AS first, max(date) AS last "
                "FROM returns GROUP BY ticker ORDER BY ticker"))
