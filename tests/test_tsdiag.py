"""Sanity checks for the time-series diagnostics.

Two ground truths drive these tests: iid returns are stationary white noise,
and a GARCH process has autocorrelated SQUARED returns (the ARCH effect).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.tsdiag import (autocorrelation, decompose, stationarity,
                               tsdiag_summary)


def _simulate_garch(n, omega, alpha, beta, seed=0) -> pd.Series:
    """GARCH(1,1) path: white-noise returns but clustered (autocorrelated) vol."""
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        r = np.sqrt(sigma2) * rng.normal()
        out[i] = r
        sigma2 = omega + alpha * r ** 2 + beta * sigma2
    return pd.Series(out, index=pd.bdate_range("2012-01-01", periods=n))


def test_adf_returns_stationary_but_level_is_not():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 1500),
                  index=pd.bdate_range("2015-01-01", periods=1500))
    st = stationarity(r)
    assert st["returns"]["adf"]["stationary"] is True    # iid returns reject unit root
    assert st["level"]["adf"]["stationary"] is False     # wealth index wanders


def test_kpss_agrees_returns_are_stationary():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 1500))
    assert stationarity(r)["returns"]["kpss"]["stationary"] is True


def test_ljung_box_flags_arch_on_vol_clustering():
    s = _simulate_garch(3000, 1e-5, 0.08, 0.90, seed=3)
    ac = autocorrelation(s)
    assert ac["ljung_box_squared"]["arch_effects"] is True   # squared returns cluster


def test_ljung_box_no_arch_on_iid_returns():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 3000))
    ac = autocorrelation(r)
    assert ac["ljung_box_squared"]["arch_effects"] is False
    assert ac["ljung_box_returns"]["white_noise"] is True


def test_acf_pacf_shapes_and_lag0():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 800))
    ac = autocorrelation(r, nlags=30)
    assert len(ac["acf"]) == 31 and len(ac["pacf"]) == 31
    assert ac["acf"][0] == pytest.approx(1.0)            # lag-0 autocorrelation is 1
    assert len(ac["acf_confint"]) == 31


def test_decompose_returns_aligned_components():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0, 0.01, 600),
                  index=pd.bdate_range("2016-01-01", periods=600))
    d = decompose(r, period=21)
    n = len(d["observed"])
    assert n > 0
    assert len(d["trend"]) == len(d["seasonal"]) == len(d["resid"]) == len(d["index"]) == n


def test_summary_has_both_blocks():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("ABC"),
                      index=pd.bdate_range("2018-01-01", periods=500))
    s = tsdiag_summary(df)
    assert {"as_of", "stationarity", "autocorrelation"} <= set(s)
