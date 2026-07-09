"""Checks for the decision layer: tactical backtest, analog days,
semi-supervised candidates, and covariance denoising — hand oracles throughout.

  • denoising     — on a known 1-factor world the denoised covariance must sit
                    CLOSER to the true covariance than the raw sample one
  • analogs       — forward returns recomputed by hand must match; embargoed
                    days must never appear as analogs
  • semi-sup      — a crash day deliberately left unflagged must be recovered
                    as a candidate from three labelled crash days
  • tactical      — with the gate never firing the strategy IS the aggressive
                    book; with it always firing the regime is 100% defensive
"""
import numpy as np
import pandas as pd
import pytest

import src.models.tactical as tactical
from src.models.analogs import EMBARGO_DAYS, FORWARD_DAYS, analog_days
from src.models.semisup import propagate_labels
from src.models.tactical import compare_overlays, walk_forward_backtest
from src.models.unsupervised import denoised_covariance, denoising_backtest


def _one_factor(n_assets=10, T=1200, seed=3, factor_vol=0.01, noise_vol=0.004):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, factor_vol, T)
    betas = rng.uniform(0.5, 1.5, n_assets)
    data = {f"A{i}": betas[i] * f + rng.normal(0, noise_vol, T)
            for i in range(n_assets)}
    true_cov = (np.outer(betas, betas) * factor_vol ** 2
                + np.eye(n_assets) * noise_vol ** 2) * 252
    return pd.DataFrame(data, index=pd.bdate_range("2019-01-01", periods=T)), true_cov


# --- covariance denoising ------------------------------------------------------
def test_denoised_covariance_closer_to_truth_when_estimation_is_noisy():
    # denoising earns its keep when N/T is sizable (sample cov is noisy);
    # with N=40 assets and T=160 days the raw estimate carries real MP noise
    r, true_cov = _one_factor(n_assets=40, T=160)
    cov_d, info = denoised_covariance(r)
    cov_raw = r.cov().to_numpy() * 252
    err_raw = np.linalg.norm(cov_raw - true_cov)
    err_den = np.linalg.norm(cov_d.to_numpy() - true_cov)
    assert err_den < err_raw                       # cleaning removed noise, not signal
    assert info["n_signal"] >= 1                   # the factor eigenvalue survives
    # diagonal (variances) is untouched by construction
    np.testing.assert_allclose(np.diag(cov_d), np.diag(cov_raw), rtol=1e-8)


def test_pure_noise_denoises_to_the_identity_correlation():
    # true correlation of independent series IS the identity; flattening every
    # below-cutoff eigenvalue must recover it almost exactly
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (800, 8)), columns=list("ABCDEFGH"))
    cov_d, info = denoised_covariance(r)
    corr_d = cov_d.to_numpy() / np.outer(r.std(), r.std()) / 252
    assert info["n_signal"] == 0
    np.testing.assert_allclose(corr_d, np.eye(8), atol=1e-9)


def test_denoising_backtest_shapes():
    r, _ = _one_factor(T=800)
    out = denoising_backtest(r)
    assert set(out["realized_vol"]) == {"raw_min_variance",
                                        "denoised_min_variance", "equal_weight"}
    assert all(v > 0 for v in out["realized_vol"].values())
    assert out["weights"]["denoised"].sum() == pytest.approx(1.0, abs=1e-6)


# --- analog days ---------------------------------------------------------------
def test_analog_forward_returns_match_hand_computation():
    r, _ = _one_factor(T=600, seed=9)
    out = analog_days(r)
    port = r.mean(axis=1)                          # equal-weight portfolio return
    pos = {d: i for i, d in enumerate(port.index)}
    top_date = out["analogs"].index[0]
    p = pos[top_date]
    fwd = float(np.prod(1 + port.iloc[p + 1:p + 1 + FORWARD_DAYS]) - 1)
    assert out["analogs"].loc[top_date, "fwd_21d"] == pytest.approx(fwd, abs=1e-3)


def test_analogs_respect_embargo_and_sorting():
    r, _ = _one_factor(T=600, seed=9)
    out = analog_days(r)
    cutoff = r.index[-(EMBARGO_DAYS + FORWARD_DAYS)]
    assert (out["analogs"].index < cutoff).all()   # nothing embargoed sneaks in
    d = out["analogs"]["distance"].to_numpy()
    assert (np.diff(d) >= 0).all()                 # nearest first
    assert out["k"] == len(out["paths"])


# --- semi-supervised candidates -------------------------------------------------
def test_propagation_recovers_the_unflagged_crash_day():
    rng = np.random.default_rng(5)
    T = 500
    base = rng.normal(0.0003, 0.006, T)
    crash_days = [100, 200, 300, 400]
    for c in crash_days:
        base[c] = -0.08                            # four identical crash days
    r = pd.DataFrame({f"A{i}": base + rng.normal(0, 0.001, T) for i in range(6)},
                     index=pd.bdate_range("2020-01-01", periods=T))

    # fake detector output: three crashes confirmed, the fourth deliberately
    # missed. Scores must VARY across ordinary days — the normal-seed picker is
    # a rank quantile, and constant scores would tie every day into the bottom
    # bucket (seeding the missed crash as "normal" and clamping it there).
    anoms = pd.DataFrame({"if_score": rng.uniform(0.1, 0.5, T),
                          "ae_error": rng.uniform(0.0, 0.3, T),
                          "if_flag": False, "ae_flag": False, "both_flag": False},
                         index=r.index)
    anoms.iloc[crash_days[3], :2] = [0.6, 0.4]     # near-miss: high-ish, unflagged
    for c in crash_days[:3]:
        anoms.iloc[c] = {"if_score": 0.9, "ae_error": 0.9,
                         "if_flag": True, "ae_flag": True, "both_flag": True}

    # 3 neighbours: the missed crash's nearest days are the three confirmed ones
    out = propagate_labels(r, anoms, n_neighbors=3)
    missed = r.index[crash_days[3]]
    calm = r.index[250]
    assert out.loc[missed, "candidate"]            # the near-miss is recovered
    assert out.loc[missed, "anomaly_prob"] > out.loc[calm, "anomaly_prob"]
    assert not out.loc[calm, "candidate"]


# --- tactical backtest -----------------------------------------------------------
def _calm_returns(T=520, n=4, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0006, 0.004, (T, n)),
                        columns=list("WXYZ"),
                        index=pd.bdate_range("2022-01-01", periods=T))


def test_gate_never_fires_means_strategy_is_the_aggressive_book():
    # calm gaussian world: no 5% forward drawdowns -> P(stress)=base rate 0
    r = _calm_returns()
    out = walk_forward_backtest(r, min_train_days=300)
    assert out["pct_defensive"] == 0.0
    pd.testing.assert_series_equal(out["daily"]["tactical"],
                                   out["daily"]["always_max_sharpe"],
                                   check_names=False)


def test_gate_always_fires_means_fully_defensive(monkeypatch):
    r = _calm_returns(seed=4)
    monkeypatch.setattr(tactical, "_stress_probability", lambda h: 1.0)
    out = walk_forward_backtest(r, min_train_days=300)
    assert out["pct_defensive"] == 1.0
    assert out["rebalance_log"]["defensive"].all()


def test_backtest_is_out_of_sample_and_well_formed():
    r = _calm_returns(T=560, seed=7)
    out = walk_forward_backtest(r, min_train_days=300)
    # first strategy day is strictly after the training warm-up
    assert out["daily"].index[0] == r.index[301]
    assert set(out["stats"].index) == {"tactical", "always_max_sharpe",
                                       "equal_weight"}
    assert out["n_rebalances"] == len(out["rebalance_log"])
    assert len(out["daily"]) == out["test_days"]


def test_compare_overlays_merges_both_variants():
    r = _calm_returns(T=520, seed=8)
    out = compare_overlays(r, min_train_days=300)
    assert set(out["daily"].columns) == {"tactical_max_sharpe",
                                         "always_max_sharpe",
                                         "tactical_equal_weight", "equal_weight"}
    assert set(out["stats"].index) == set(out["daily"].columns)
