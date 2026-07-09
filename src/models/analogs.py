"""Analog days — "when did the market last look like today, and what happened
next?"

K-nearest-neighbours over the same six market-state features the stress
classifier uses (vol, EWMA vol, drawdown, momentum, dispersion, last return).
Instead of fitting a model, KNN just retrieves precedent: the k most similar
historical days, each with the forward portfolio return that actually followed.
The distribution of those forward paths is an empirical, assumption-free
answer to "given days like today, what tended to happen?" — the desk habit of
reasoning by market analogs, made systematic.

Recent days are embargoed from the candidate pool: yesterday is trivially the
most similar day to today and carries no independent information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.classify import build_features
from src.models.risk import portfolio_returns

# why 20: enough analogs for a readable forward distribution, few enough that
# each is genuinely similar to today rather than just "any old day".
N_ANALOGS = 20

# why 21: exclude the trailing month — those days share today's rolling
# windows, so they'd match on overlap rather than genuine similarity.
EMBARGO_DAYS = 21

# why 21: report the forward month, matching the classifier's stress horizon.
FORWARD_DAYS = 21


def analog_days(returns: pd.DataFrame,
                k: int = N_ANALOGS,
                embargo: int = EMBARGO_DAYS,
                forward: int = FORWARD_DAYS) -> dict:
    """Find the k historical days most similar to the LATEST day.

    Returns the analog dates with their similarity distance and realised
    forward 5d/21d portfolio returns, summary quantiles of the forward-month
    distribution, and each analog's cumulative forward path (for a fan chart).
    Analogs too close to the end of history to have a full forward window are
    excluded — every reported outcome is fully realised.
    """
    feats = build_features(returns).dropna()
    port = portfolio_returns(returns)

    query = feats.iloc[[-1]]
    query_date = feats.index[-1]
    # candidates: full forward window available AND outside the embargo
    pool = feats.iloc[:-(embargo + forward)]

    scaler = StandardScaler().fit(pool)
    nn = NearestNeighbors(n_neighbors=min(k, len(pool)))
    nn.fit(scaler.transform(pool))
    dist, idx = nn.kneighbors(scaler.transform(query))

    rows, paths = [], {}
    pos = {d: i for i, d in enumerate(port.index)}
    pvals = port.to_numpy()
    for d, i in zip(dist[0], idx[0]):
        date = pool.index[i]
        p = pos[date]
        fwd = pvals[p + 1:p + 1 + forward]
        path = np.cumprod(1 + fwd) - 1
        rows.append({"date": date, "distance": round(float(d), 3),
                     "fwd_5d": round(float(path[4]), 4),
                     "fwd_21d": round(float(path[-1]), 4)})
        paths[str(date.date())] = [round(float(v), 4) for v in path]

    analogs = pd.DataFrame(rows).set_index("date").sort_values("distance")
    fwd = analogs["fwd_21d"]
    return {
        "query_date": str(query_date.date()),
        "k": len(analogs),
        "analogs": analogs,
        "paths": paths,          # per-analog cumulative forward path (fan chart)
        "summary": {
            "median_fwd_21d": round(float(fwd.median()), 4),
            "p10_fwd_21d": round(float(fwd.quantile(0.10)), 4),
            "p90_fwd_21d": round(float(fwd.quantile(0.90)), 4),
            "pct_negative": round(float((fwd < 0).mean()), 4),
        },
    }


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide

    ensure_loaded()
    out = analog_days(returns_wide())
    s = out["summary"]
    print(f"{out['k']} nearest analogs to {out['query_date']} -> forward month: "
          f"median {s['median_fwd_21d']:+.1%}, "
          f"p10 {s['p10_fwd_21d']:+.1%}, p90 {s['p90_fwd_21d']:+.1%}, "
          f"{s['pct_negative']:.0%} ended down")
    print(out["analogs"].to_string())
