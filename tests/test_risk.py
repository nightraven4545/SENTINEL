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


# ---------------------------------------------------------------- deep-dive metrics

from src.models.risk import (TRADING_DAYS as T, _DAILY_RF, capm, calmar_ratio,
                             cornish_fisher_var, drawdown_series,
                             expected_shortfall, kupiec_test,
                             max_drawdown_duration, risk_contributions,
                             portfolio_returns as pr, annualized_vol,
                             sharpe_ratio, sortino_ratio)


@pytest.fixture
def normal_returns() -> pd.Series:
    rng = np.random.default_rng(5)
    return series(rng.normal(0.0004, 0.01, 20_000))


def test_expected_shortfall_exceeds_var(normal_returns):
    assert expected_shortfall(normal_returns, 0.95) > hist_var(normal_returns, 0.95)
    assert expected_shortfall(normal_returns, 0.99) > hist_var(normal_returns, 0.99)


def test_expected_shortfall_known_values():
    # 100 equally likely returns -1%..-100%: worst-5% tail mean ≈ (96..100)/100
    r = series([-i / 100 for i in range(1, 101)])
    assert expected_shortfall(r, 0.95) == pytest.approx(0.98, abs=0.011)


def test_cornish_fisher_matches_normal_var_on_gaussian(normal_returns):
    # zero skew/kurtosis -> CF collapses to plain parametric VaR ≈ historical
    cf = cornish_fisher_var(normal_returns, 0.95)
    assert cf == pytest.approx(hist_var(normal_returns, 0.95), rel=0.05)


def test_cornish_fisher_penalizes_negative_skew():
    rng = np.random.default_rng(9)
    base = rng.normal(0, 0.01, 10_000)
    skewed = np.where(rng.random(10_000) < 0.03, base - 0.04, base)  # left tail
    assert cornish_fisher_var(series(skewed), 0.99) > cornish_fisher_var(
        series(base), 0.99)


def test_sharpe_ratio_known_input(normal_returns):
    mu, sd = normal_returns.mean(), normal_returns.std()
    expected = (mu - _DAILY_RF) / sd * np.sqrt(T)
    assert sharpe_ratio(normal_returns) == pytest.approx(expected)


def test_sortino_uses_downside_only():
    # upside-heavy series: sortino must exceed sharpe (denominator shrinks)
    rng = np.random.default_rng(2)
    r = series(np.abs(rng.normal(0, 0.01, 5_000)) - 0.002)
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_calmar_ratio_known_input():
    r = series([1.0, -0.5] * 5)  # ann_return known, max_dd = -0.5
    expected = (r.mean() * T) / 0.5
    assert calmar_ratio(r) == pytest.approx(expected)


def test_capm_recovers_planted_beta():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=3_000)
    bench = pd.Series(rng.normal(0.0004, 0.01, 3_000), index=idx)
    port = 1.5 * bench + rng.normal(0, 0.001, 3_000)
    stats = capm(port, bench)
    assert stats["beta"] == pytest.approx(1.5, abs=0.02)
    assert stats["r2"] > 0.95
    assert stats["tracking_error"] > 0


def test_risk_contributions_euler_identity():
    rng = np.random.default_rng(4)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, (2_000, 4)), columns=list("ABCD"))
    r["A"] *= 3  # one deliberately risky name
    contrib = risk_contributions(r)
    # contributions sum exactly to portfolio vol; percentages to 1
    assert contrib["risk_contribution"].sum() == pytest.approx(
        float(annualized_vol(pr(r))))
    assert contrib["pct_of_risk"].sum() == pytest.approx(1.0)
    # the risky name carries more risk than capital
    assert contrib.loc["A", "pct_of_risk"] > contrib.loc["A", "weight"]


def test_kupiec_accepts_calibrated_var():
    # realistic sample length matters: rolling-quantile VaR breaches slightly
    # above nominal (estimator noise), which huge samples would flag — at a
    # real desk history (~2.5k days) a calibrated model is not rejected.
    rng = np.random.default_rng(5)
    bt = kupiec_test(series(rng.normal(0.0004, 0.01, 2_500)), 0.95)
    assert bt["passes"] is True
    assert bt["observations"] == 2_500 - 250
    assert 0.03 < bt["breach_rate"] < 0.07


def test_kupiec_rejects_broken_var():
    # regime change the trailing window can't see: vol quadruples halfway
    rng = np.random.default_rng(6)
    calm = rng.normal(0, 0.005, 2_000)
    wild = rng.normal(0, 0.02, 500)
    bt = kupiec_test(series(np.concatenate([calm, wild])), 0.95, window=1_500)
    assert bt["breach_rate"] > 0.05  # under-forecasts risk
    assert bt["passes"] is False


def test_kupiec_handles_insufficient_history():
    bt = kupiec_test(series([0.01, -0.02, 0.0]), 0.95, window=250)  # n < window
    assert bt["observations"] == 0
    assert bt["passes"] is False  # no crash, honest "can't test"


def test_drawdown_duration_counts_underwater_days():
    # wealth: 2 -> 1 -> 2.2 : underwater exactly one day
    r = series([1.0, -0.5, 1.2])
    assert max_drawdown_duration(r) == 1
    assert drawdown_series(r).min() == pytest.approx(-0.5)


# ---------------------------------------------------------------- EWMA conditional vol

from src.models.risk import EWMA_LAMBDA, ewma_var, ewma_vol


def test_ewma_vol_constant_magnitude_returns():
    # |r| constant -> conditional variance is that constant, at every step
    r = series([0.01, -0.01] * 500)
    assert ewma_vol(r).iloc[-1] == pytest.approx(0.01 * np.sqrt(TRADING_DAYS))


def test_ewma_vol_reacts_to_a_shock():
    # a fat return spikes the conditional vol vs the calm run before it
    r = series([0.001] * 200 + [0.10] + [0.001] * 5)
    v = ewma_vol(r)
    assert v.iloc[201] > v.iloc[199] * 3  # the day after the shock


def test_ewma_var_scales_with_confidence():
    rng = np.random.default_rng(7)
    r = series(rng.normal(0, 0.01, 2_000))
    assert ewma_var(r, 0.99) > ewma_var(r, 0.95) > 0


def test_ewma_var_known_value_on_constant_vol():
    # daily sigma 0.01 -> VaR95 = |z_0.05| * 0.01
    r = series([0.01, -0.01] * 500)
    from scipy.stats import norm
    assert ewma_var(r, 0.95) == pytest.approx(abs(norm.ppf(0.05)) * 0.01, rel=1e-6)


def test_ewma_lambda_is_riskmetrics_daily():
    assert EWMA_LAMBDA == 0.94
