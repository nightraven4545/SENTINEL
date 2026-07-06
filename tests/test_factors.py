"""Checks for the OLS engine and the Fama-French factor regression.

No network: the factor frame is synthetic, and returns are built from known
betas so the regression must recover them exactly (zero-noise) or closely.
"""
import numpy as np
import pandas as pd
import pytest

from src.ingest.factors import FACTORS
from src.models.factors import _ols, factor_loadings, factor_regression


def synthetic_factors(n=400, seed=0) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({c: rng.normal(0, 0.01, n) for c in FACTORS}, index=dates)
    f["rf"] = 0.0001
    return f


def test_ols_recovers_known_coefficients_exactly_without_noise():
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=500), rng.normal(size=500)
    y = 2.0 + 3.0 * x1 - 1.5 * x2  # no noise
    X = np.column_stack([np.ones(500), x1, x2])
    res = _ols(y, X)
    assert res["beta"] == pytest.approx([2.0, 3.0, -1.5], abs=1e-9)
    assert res["r2"] == pytest.approx(1.0, abs=1e-12)


def test_ols_matches_numpy_lstsq():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(300), rng.normal(size=(300, 3))])
    y = rng.normal(size=300)
    ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert _ols(y, X)["beta"] == pytest.approx(ref, abs=1e-9)


def test_factor_regression_recovers_betas_and_alpha():
    f = synthetic_factors()
    true = {"mkt_rf": 1.1, "smb": -0.3, "hml": 0.2, "rmw": 0.0, "cma": 0.0, "mom": 0.5}
    alpha_daily = 0.0002
    excess = sum(true[c] * f[c] for c in FACTORS)
    ret = alpha_daily + excess + f["rf"]  # ret = alpha + betas·factors + rf

    r = factor_regression(ret, f)
    for c in FACTORS:
        assert r["betas"][c] == pytest.approx(true[c], abs=1e-6)
    assert r["alpha_ann"] == pytest.approx(alpha_daily * 252, abs=1e-4)
    assert r["r2"] == pytest.approx(1.0, abs=1e-9)


def test_factor_regression_aligns_on_overlap():
    f = synthetic_factors(n=300)
    # returns carry extra later dates the factor library doesn't cover -> ignored
    extra = pd.bdate_range("2021-06-01", periods=50)
    ret = pd.Series(0.001, index=f.index.append(extra))
    r = factor_regression(ret, f)
    assert r["n"] == 300  # only the overlapping dates enter the regression


def test_factor_loadings_one_row_per_ticker():
    f = synthetic_factors()
    rng = np.random.default_rng(3)
    wide = pd.DataFrame({"AAA": rng.normal(0, 0.01, len(f)),
                         "BBB": rng.normal(0, 0.01, len(f))}, index=f.index)
    load = factor_loadings(wide, f)
    assert list(load.index) == ["AAA", "BBB"]
    assert list(load.columns) == [*FACTORS, "alpha_ann", "r2"]
