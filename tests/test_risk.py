"""Known-input sanity checks for the risk metrics."""
import numpy as np
import pandas as pd
import pytest

from src.models.risk import (
    TRADING_DAYS,
    annualized_vol,
    correlation_matrix,
    hist_var,
    max_drawdown,
    portfolio_returns,
)


def series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_constant_returns_have_zero_vol_and_no_drawdown():
    r = series([0.01] * 100)
    assert annualized_vol(r) == pytest.approx(0.0)
    assert max_drawdown(r) == pytest.approx(0.0)  # wealth only ever rises


def test_annualized_vol_scaling():
    rng = np.random.default_rng(0)
    r = series(rng.normal(0, 0.01, 10_000))
    assert annualized_vol(r) == pytest.approx(r.std() * np.sqrt(TRADING_DAYS))


def test_hist_var_is_empirical_quantile_as_positive_loss():
    # 100 equally likely returns: -1%, -2%, ..., -100%.
    r = series([-i / 100 for i in range(1, 101)])
    # 5% worst tail starts around the -95% return -> VaR95 ~ +0.95.
    assert hist_var(r, 0.95) == pytest.approx(0.9505, abs=1e-3)
    assert hist_var(r, 0.99) == pytest.approx(0.9901, abs=1e-3)


def test_var_is_positive_for_losing_days():
    r = series([0.01, -0.05, 0.02, -0.03] * 50)
    assert hist_var(r, 0.95) > 0


def test_max_drawdown_double_then_halve():
    # +100% then -50%: wealth 1 -> 2 -> 1, worst peak-to-trough = -50%.
    r = series([1.0, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_portfolio_returns_equal_weight():
    r = pd.DataFrame({"A": [0.02, 0.0], "B": [0.0, -0.02]})
    port = portfolio_returns(r)
    assert port.tolist() == pytest.approx([0.01, -0.01])


def test_correlation_matrix_shape_and_diagonal():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("ABC"))
    corr = correlation_matrix(r)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
