"""Fama-French 5 factors + momentum, from Ken French's Data Library.

The academic factor returns Sentinel regresses the portfolio against to explain
*why* it earns what it does: market, size (SMB), value (HML), profitability
(RMW), investment (CMA) and momentum (MOM). Free, no key — just two CSV zips.

Returned wide (date index, one column per factor) as DECIMAL daily returns
(the library publishes percentages; we divide by 100 to match yfinance returns).
`rf` is the daily risk-free rate that goes with them.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF5 = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM = "F-F_Momentum_Factor_daily_CSV.zip"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = DATA_DIR / "factors.parquet"
# Committed snapshot so cloud demos work if Dartmouth is slow/blocked — same
# resilience pattern as prices/fundamentals.
SAMPLE_FILE = DATA_DIR / "samples" / "ff_factors_sample.parquet"

FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]  # rf handled separately
_DATA_ROW = re.compile(r"^\s*\d{8}\s*,")  # a YYYYMMDD-prefixed data line


def _download(name: str) -> str:
    req = urllib.request.Request(_BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    return zf.read(zf.namelist()[0]).decode("latin-1")


def _parse(text: str, columns: list[str]) -> pd.DataFrame:
    """Pull the daily rows out of a Ken French CSV (skipping its prose header /
    any annual footer) into a date-indexed decimal-returns frame."""
    rows = [ln for ln in text.splitlines() if _DATA_ROW.match(ln)]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     usecols=range(len(columns) + 1), names=["date", *columns])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date")[columns] / 100.0  # percent -> decimal


def fetch_factors(use_cache: bool = True) -> pd.DataFrame:
    """Daily FF5 + momentum + rf, wide. cache -> network -> committed sample."""
    if use_cache and CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)

    try:
        ff5 = _parse(_download(_FF5), ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
        mom = _parse(_download(_MOM), ["mom"])
        wide = ff5.join(mom, how="inner")[[*FACTORS, "rf"]]
    except Exception:
        wide = pd.DataFrame()

    if wide.empty:
        if SAMPLE_FILE.exists():
            return pd.read_parquet(SAMPLE_FILE)
        raise RuntimeError("Could not fetch Fama-French factors and no sample exists")

    DATA_DIR.mkdir(exist_ok=True)
    wide.to_parquet(CACHE_FILE)
    return wide


if __name__ == "__main__":
    f = fetch_factors()
    print(f"{len(f):,} days, {f.index.min():%Y-%m-%d} -> {f.index.max():%Y-%m-%d}")
    print(f.tail().round(4))
