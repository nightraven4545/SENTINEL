"""Semi-supervised anomaly hunt — label propagation from the confirmed events.

The two unsupervised detectors (IsolationForest + autoencoder) agree on a
handful of high-conviction anomaly days. Semi-supervised learning asks the
natural follow-up: *which other days look like those?*

We seed three groups over the same feature space the detectors saw:
  label 1 — days BOTH detectors flagged (the confirmed anomalies)
  label 0 — the calmest days (bottom of both detectors' score ranks)
  unlabeled — everything else (the vast majority)

and let scikit-learn's LabelSpreading diffuse the labels through the KNN graph
of trading days: a day inherits "anomalous" if its nearest neighbours in
feature space are anomalous. The output worth reading is the CANDIDATE list —
days with high propagated anomaly probability that NEITHER detector flagged:
the near-misses sitting just under both models' thresholds.
"""
from __future__ import annotations

import pandas as pd
from sklearn.semi_supervised import LabelSpreading

from src.models.anomaly import build_features, detect_anomalies

# why 0.30: the bottom 30% of days by combined detector rank are safely
# "normal" seeds — far from any threshold, so mislabelling risk is negligible.
NORMAL_SEED_FRAC = 0.30

# why 10: KNN kernel over ~2k trading days — enough neighbours to smooth the
# graph, few enough that labels don't bleed across unrelated regimes.
KNN_NEIGHBORS = 10

def propagate_labels(returns: pd.DataFrame,
                     anomalies: pd.DataFrame | None = None,
                     n_neighbors: int = KNN_NEIGHBORS) -> pd.DataFrame:
    """Spread anomaly labels from the confirmed days through feature space.

    Returns one row per day: propagated anomaly probability, the seed label it
    started with (1 / 0 / -1 = unlabeled), the detector flags, and `candidate`.

    The candidate threshold is self-calibrating rather than a magic number:
    a day is a candidate when it is NOT already a confirmed (both-detector)
    anomaly but propagation scores it at least as anomalous as the WEAKEST
    confirmed event — "if that day counted, these should have too". With 14
    positive seeds against hundreds of negatives, a fixed 0.5 posterior would
    never fire; anchoring on the confirmed events keeps the bar meaningful.
    """
    feats = build_features(returns)
    if anomalies is None:
        anomalies = detect_anomalies(returns)
    anomalies = anomalies.loc[feats.index]

    # seed labels: confirmed anomalies = 1, calmest days = 0, rest unlabeled
    combined_rank = (anomalies["if_score"].rank() + anomalies["ae_error"].rank())
    labels = pd.Series(-1, index=feats.index, dtype=int)
    labels[combined_rank <= combined_rank.quantile(NORMAL_SEED_FRAC)] = 0
    labels[anomalies["both_flag"]] = 1

    model = LabelSpreading(kernel="knn", n_neighbors=n_neighbors)
    model.fit(feats.to_numpy(), labels.to_numpy())
    prob = model.label_distributions_[:, list(model.classes_).index(1)]

    out = pd.DataFrame({
        "anomaly_prob": prob.round(4),
        "seed": labels,
        "if_flag": anomalies["if_flag"],
        "ae_flag": anomalies["ae_flag"],
        "both_flag": anomalies["both_flag"],
    }, index=feats.index)
    # bar = the weakest confirmed anomaly's propagated probability
    bar = float(out.loc[out["both_flag"], "anomaly_prob"].min())
    out["candidate"] = (out["anomaly_prob"] >= bar) & ~out["both_flag"]
    return out


def anomaly_candidates(returns: pd.DataFrame,
                       anomalies: pd.DataFrame | None = None,
                       top_n: int = 10) -> pd.DataFrame:
    """The ranked near-misses: days scoring at least as anomalous as the
    weakest confirmed event but not confirmed themselves. Includes the
    detector-DISAGREEMENT days (one flag, not both) that propagation sides
    with — the adjudicated near-misses — plus any day both models let through.
    """
    out = propagate_labels(returns, anomalies)
    cands = out[out["candidate"]].sort_values("anomaly_prob", ascending=False)
    return cands.head(top_n)[["anomaly_prob", "if_flag", "ae_flag"]]


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide

    ensure_loaded()
    r = returns_wide()
    out = propagate_labels(r)
    seeds = int((out["seed"] == 1).sum())
    cands = out[out["candidate"]]
    print(f"{len(out)} days, {seeds} confirmed-anomaly seeds, "
          f"{int((out['seed'] == 0).sum())} normal seeds -> "
          f"{len(cands)} near-miss candidates (as anomalous as the weakest "
          f"confirmed event, not confirmed themselves):")
    print(anomaly_candidates(r).to_string())
