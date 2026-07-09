"""Unsupervised structure — the course's W3 lens (PCA + KMeans clustering).

Two questions no single-name metric answers:

  • pca_factors  — how many INDEPENDENT sources of risk actually drive the
    portfolio? PCA on the standardized returns extracts orthogonal statistical
    factors; the first (PC1) is almost always the market itself. This is the
    *statistical* complement to factors.py's *economic* Fama-French factors:
    Fama-French names its factors up front, PCA lets the data name them.

  • cluster_universe — which names behave alike? KMeans groups the tickers by
    their risk/return fingerprint, with the silhouette score choosing k. It's an
    independent cross-check on graph.py's correlation-network communities: two
    different algorithms should recover the same sector blocks.

Both are sklearn — no new dependency. PCA uses the CORRELATION matrix (returns
standardized first) so that a single high-variance name can't masquerade as a
common factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.models.risk import (TRADING_DAYS, annualized_return, annualized_vol,
                             max_drawdown, portfolio_returns)

# why up to 3: PC1 is the market, PC2/PC3 the leading style rotations — beyond
# that the components are mostly idiosyncratic noise for a ~10-name book.
N_COMPONENTS = 3

# why 2..5: with ~10 tickers, fewer than 2 clusters is trivial and more than 5
# splits sectors into singletons. Silhouette picks the best k in this range.
K_RANGE = range(2, 6)

RANDOM_STATE = 42


def pca_factors(returns: pd.DataFrame, n_components: int = N_COMPONENTS) -> dict:
    """Principal-component analysis of the (standardized) daily returns.

    Standardizing first makes this a PCA of the CORRELATION matrix: every name
    contributes unit variance, so PC1 captures shared co-movement rather than
    just the loudest ticker. Returns the variance each component explains and the
    per-ticker loadings (how each name projects onto each factor).
    """
    clean = returns.dropna()
    z = StandardScaler().fit_transform(clean)
    k = min(n_components, clean.shape[1])
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    pca.fit(z)

    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=[f"PC{i + 1}" for i in range(k)],
    )
    return {
        "explained_variance_ratio": evr,
        "cumulative_variance": [round(float(v), 4) for v in np.cumsum(evr)],
        "loadings": loadings,
        "n_obs": int(len(clean)),
        "n_assets": int(clean.shape[1]),
    }


def risk_return_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker risk/return fingerprint — for the dashboard scatter/table, not
    for clustering (magnitude features let one outlier form its own cluster)."""
    port = portfolio_returns(returns)
    betas = {t: float(returns[t].cov(port) / port.var()) for t in returns.columns}
    return pd.DataFrame({
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "max_drawdown": returns.apply(max_drawdown),
        "beta": pd.Series(betas),
    })


def cluster_universe(returns: pd.DataFrame, k_range=K_RANGE) -> dict:
    """KMeans-cluster the tickers by their CORRELATION profile.

    Each ticker's feature vector is its row of the correlation matrix — who it
    co-moves with. Clustering on co-movement (not on return/vol magnitude)
    recovers sector blocks and makes this a like-for-like cross-check on
    graph.py's correlation-network communities. Sweeps k over `k_range` and
    keeps the labeling with the best silhouette score (tightness/separation in
    [-1, 1]).
    """
    feats = returns.corr()  # each row: a name's co-movement fingerprint
    z = StandardScaler().fit_transform(feats)

    scored = []
    for k in k_range:
        if k >= len(feats):  # need at least k+1 points for a meaningful silhouette
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(z)
        scored.append((float(silhouette_score(z, labels)), k, labels))

    best_sil, best_k, best_labels = max(scored, key=lambda t: t[0])
    return {
        "k": int(best_k),
        "silhouette": round(best_sil, 4),
        "labels": {t: int(c) for t, c in zip(feats.index, best_labels)},
        "silhouette_by_k": {int(k): round(s, 4) for s, k, _ in scored},
    }


def denoised_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Random-matrix (Marchenko-Pastur) denoising of the covariance matrix.

    A sample correlation matrix estimated from T days of N assets carries
    measurement noise with a KNOWN eigenvalue spectrum: pure noise would put
    every eigenvalue below lambda+ = (1 + sqrt(N/T))^2 (Marchenko-Pastur).
    Eigenvalues above the cutoff are signal (market, sectors); the rest are
    noise the Markowitz optimiser would happily lever up. We keep the signal
    eigenvalues, flatten the noisy ones to their average (preserving the
    trace), rebuild the correlation matrix and re-apply the sample vols.

    Returns the annualized denoised covariance plus diagnostics (eigenvalues,
    the cutoff, how many eigenvalues counted as signal).
    """
    clean = returns.dropna()
    T, N = clean.shape
    corr = clean.corr()
    std = clean.std()

    evals, evecs = np.linalg.eigh(corr.to_numpy())   # ascending order
    mp_cutoff = (1 + np.sqrt(N / T)) ** 2
    noise = evals < mp_cutoff
    denoised = evals.copy()
    if noise.any():
        denoised[noise] = evals[noise].mean()        # flatten noise, keep trace

    corr_d = evecs @ np.diag(denoised) @ evecs.T
    d = np.sqrt(np.diag(corr_d))
    corr_d = corr_d / np.outer(d, d)                 # unit diagonal again
    cov_d = corr_d * np.outer(std, std) * TRADING_DAYS

    return (pd.DataFrame(cov_d, index=corr.index, columns=corr.columns),
            {"eigenvalues": [round(float(v), 4) for v in evals[::-1]],
             "mp_cutoff": round(float(mp_cutoff), 4),
             "n_signal": int((~noise).sum()), "n_assets": N, "n_obs": T})


def denoising_backtest(returns: pd.DataFrame, split: float = 0.5) -> dict:
    """Does denoising help OUT OF SAMPLE? Estimate min-variance weights on the
    first part of history with the raw vs denoised covariance, then measure the
    realized volatility of both books on the unseen second part. Lower realized
    vol from the denoised weights = the cleaning removed noise, not signal.
    """
    from src.models.optimize import min_variance
    cut = int(len(returns) * split)
    est, test = returns.iloc[:cut], returns.iloc[cut:]

    cov_d, info = denoised_covariance(est)
    w_raw = min_variance(est)
    w_den = min_variance(est, cov=cov_d)
    w_eq = pd.Series(1.0 / returns.shape[1], index=returns.columns)

    realized = {name: float(annualized_vol(test @ w.reindex(test.columns)))
                for name, w in
                (("raw_min_variance", w_raw), ("denoised_min_variance", w_den),
                 ("equal_weight", w_eq))}
    return {"realized_vol": {k: round(v, 4) for k, v in realized.items()},
            "weights": pd.DataFrame({"raw": w_raw, "denoised": w_den}),
            "info": info,
            "est_days": int(cut), "test_days": int(len(test))}


def compare_to_communities(returns: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side of the KMeans clusters and graph.py's network communities —
    two unsupervised methods on the same universe. Agreement is the finding:
    both should recover the same sector blocks from different math."""
    from src.models.graph import build_graph, communities
    km = cluster_universe(returns)["labels"]
    comm = communities(build_graph(returns.corr()))
    return pd.DataFrame({
        "kmeans_cluster": pd.Series(km),
        "graph_community": pd.Series(comm),
    }).sort_values("kmeans_cluster")


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide

    ensure_loaded()
    r = returns_wide()

    pca = pca_factors(r)
    print(f"PCA on {pca['n_assets']} names, {pca['n_obs']} days:")
    for i, (evr, cum) in enumerate(zip(pca["explained_variance_ratio"],
                                       pca["cumulative_variance"]), start=1):
        print(f"  PC{i}: {evr:.1%} of variance  (cumulative {cum:.1%})")
    print("\n  Loadings:")
    print(pca["loadings"].round(3))

    cl = cluster_universe(r)
    print(f"\nKMeans: best k={cl['k']} (silhouette {cl['silhouette']:.3f})")
    print("\nKMeans clusters vs graph communities:")
    print(compare_to_communities(r))

    dn = denoising_backtest(r)
    print(f"\nDenoising backtest (est {dn['est_days']}d -> test {dn['test_days']}d, "
          f"{dn['info']['n_signal']}/{dn['info']['n_assets']} signal eigenvalues, "
          f"MP cutoff {dn['info']['mp_cutoff']}):")
    for k, v in dn["realized_vol"].items():
        print(f"  {k:22} realized vol {v:.2%}")
