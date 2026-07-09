"""Checks for the course-ML lenses: the supervised stress classifier (W2/W4)
and the unsupervised PCA + KMeans structure (W3), with hand-built oracles.

The oracles are constructed so the right answer is known up front:
  • a single crash day makes every preceding in-horizon day a labelled stress day
  • one dominant common factor makes PCA's PC1 explain almost all variance
  • two correlation blocks make KMeans recover exactly those two groups
  • a vol-precedes-drawdown signal makes the classifier beat a coin flip
"""
import numpy as np
import pandas as pd
import pytest

from src.models.classify import (STRESS_DD_THRESHOLD, STRESS_FWD_DAYS,
                                  _forward_stress_label, build_dataset,
                                  train_stress_classifier)
from src.models.unsupervised import cluster_universe, pca_factors


# --- helpers -----------------------------------------------------------------
def _one_factor_returns(n_assets=5, T=500, seed=0) -> pd.DataFrame:
    """All assets = one shared factor + small idiosyncratic noise -> PC1 dominates."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, T)
    cols = {f"A{i}": common + rng.normal(0, 0.001, T) for i in range(n_assets)}
    return pd.DataFrame(cols)


def _two_block_returns(T=600, seed=1) -> pd.DataFrame:
    """Two independent factors drive two groups of three names -> two corr blocks."""
    rng = np.random.default_rng(seed)
    fa, fb = rng.normal(0, 0.01, T), rng.normal(0, 0.01, T)
    cols = {}
    for i in range(3):
        cols[f"A{i}"] = fa + rng.normal(0, 0.001, T)
        cols[f"B{i}"] = fb + rng.normal(0, 0.001, T)
    return pd.DataFrame(cols)


# --- stress-label oracle -----------------------------------------------------
def test_forward_label_marks_days_before_a_crash():
    # flat except a single -10% day at index 10; horizon 21, threshold 5%.
    port = pd.Series(np.zeros(60))
    port.iloc[10] = -0.10
    y = _forward_stress_label(port)
    # every day whose forward window still contains the crash is stress=1...
    assert y.iloc[9] == 1
    assert y.iloc[max(0, 10 - STRESS_FWD_DAYS)] == 1
    # ...the crash day itself looks forward at flat returns, so it is not
    assert y.iloc[10] == 0
    assert y.iloc[11] == 0


def test_label_threshold_respected():
    # a 3% dip never crosses the 5% threshold -> no positive labels
    port = pd.Series(np.zeros(60))
    port.iloc[10] = -0.03
    assert _forward_stress_label(port).sum() == 0
    assert STRESS_DD_THRESHOLD == 0.05


# --- dataset hygiene (no look-ahead) -----------------------------------------
def test_build_dataset_shape_and_alignment():
    r = _one_factor_returns(T=400)
    X, y = build_dataset(r)
    assert list(X.columns) == ["ret_1d", "vol_21", "ewma_vol",
                               "drawdown", "mom_21", "dispersion"]
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
    # the incomplete-horizon tail is dropped, so X is shorter than the raw history
    assert len(X) < len(r)


# --- classifier learns a real signal -----------------------------------------
def test_classifier_beats_random_on_a_learnable_signal():
    # Build vol-clustered returns: calm baseline punctuated by crash episodes
    # that are preceded by a spike in volatility. vol_21 should then predict the
    # forward drawdown, so a fitted model must clear 0.5 AUC out-of-sample.
    rng = np.random.default_rng(7)
    T = 1600
    base = rng.normal(0.0003, 0.008, T)
    for start in range(120, T - 40, 160):     # regular crash episodes
        base[start - 5:start] += rng.normal(0, 0.03, 5)   # pre-crash vol spike
        base[start:start + 15] -= 0.010                    # the drawdown itself
    # 10 correlated names around the common path
    cols = {f"T{i}": base + rng.normal(0, 0.002, T) for i in range(10)}
    out = train_stress_classifier(pd.DataFrame(cols))

    assert 0 < out["positive_rate"] < 1
    best_auc = max(m["roc_auc"] for m in out["models"].values())
    assert best_auc > 0.6
    # vol features should carry meaningful importance
    imp = out["feature_importance"]
    assert imp["vol_21"] + imp["ewma_vol"] > 0.2


# --- PCA oracle --------------------------------------------------------------
def test_pca_pc1_dominates_with_one_common_factor():
    out = pca_factors(_one_factor_returns())
    assert out["explained_variance_ratio"][0] > 0.8      # one factor => PC1 huge
    assert out["cumulative_variance"][-1] <= 1.0 + 1e-9
    # loadings: one row per asset, up to N_COMPONENTS columns
    assert out["loadings"].shape[0] == out["n_assets"]
    # PC1 is the market factor: every name loads with the same sign
    pc1 = out["loadings"]["PC1"]
    assert (pc1 > 0).all() or (pc1 < 0).all()


# --- KMeans oracle -----------------------------------------------------------
def test_kmeans_recovers_two_correlation_blocks():
    out = cluster_universe(_two_block_returns(), k_range=range(2, 4))
    assert out["k"] == 2
    assert out["silhouette"] > 0.5
    labels = out["labels"]
    a = {labels[c] for c in labels if c.startswith("A")}
    b = {labels[c] for c in labels if c.startswith("B")}
    assert len(a) == 1 and len(b) == 1 and a != b   # clean, disjoint groups
