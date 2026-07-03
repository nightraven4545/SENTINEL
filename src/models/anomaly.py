"""Day-level market anomaly detection: IsolationForest baseline + a small
PyTorch autoencoder.

Both models see the same feature matrix — one row per trading day:
each ticker's daily return and its 21d rolling vol, z-scored. A day is
"anomalous" when the cross-section of returns/vol looks unlike normal
market behaviour (crash days, vol spikes, weird dispersion).

Two models on purpose: IsolationForest is the robust, no-training-drama
baseline; the autoencoder learns what "normal" days look like and flags
days it cannot reconstruct — they usually agree on the big events, and
disagreement itself is informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest

from src.models.risk import ROLLING_WINDOW, rolling_vol

# why 0.02: ~5 flagged days a year — a sensible base rate for "days a risk
# analyst should have looked at" without drowning the user in alerts.
CONTAMINATION = 0.02
AE_THRESHOLD_PCT = 100 * (1 - CONTAMINATION)  # match the IF base rate

SEED = 42
AE_EPOCHS = 300
AE_HIDDEN = 8
AE_CODE = 3  # bottleneck: forces the net to learn structure, not memorise
AE_LR = 1e-2


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-day feature matrix: [returns..., rolling vols...], z-scored.

    Z-scoring matters because returns (~1e-2) and vols (~1e-1) live on
    different scales, and both models are scale-sensitive.
    """
    vols = rolling_vol(returns, ROLLING_WINDOW)
    feats = pd.concat(
        [returns.add_suffix("_ret"), vols.add_suffix("_vol")], axis=1
    ).dropna()
    return (feats - feats.mean()) / feats.std()


def isolation_forest_scores(features: pd.DataFrame,
                            contamination: float = CONTAMINATION) -> pd.DataFrame:
    """IsolationForest anomaly scores. Higher score = more anomalous
    (isolated with fewer random splits)."""
    model = IsolationForest(contamination=contamination, random_state=SEED)
    model.fit(features)
    # score_samples: higher = more normal, so negate for "higher = weirder".
    scores = -model.score_samples(features)
    flags = model.predict(features) == -1
    return pd.DataFrame({"if_score": scores, "if_flag": flags}, index=features.index)


class _AutoEncoder(torch.nn.Module):
    """Tiny symmetric MLP: n -> 8 -> 3 -> 8 -> n."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, AE_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(AE_HIDDEN, AE_CODE), torch.nn.ReLU(),
            torch.nn.Linear(AE_CODE, AE_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(AE_HIDDEN, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def autoencoder_scores(features: pd.DataFrame,
                       epochs: int = AE_EPOCHS,
                       threshold_pct: float = AE_THRESHOLD_PCT) -> pd.DataFrame:
    """Train on ALL days (anomalies are rare enough not to poison the fit),
    then flag days whose reconstruction error is in the top tail.

    The dataset is tiny (~2k rows), so full-batch training for a few hundred
    epochs is simpler and more reproducible than mini-batch machinery.
    """
    torch.manual_seed(SEED)
    x = torch.tensor(features.to_numpy(), dtype=torch.float32)
    model = _AutoEncoder(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        errors = ((model(x) - x) ** 2).mean(dim=1).numpy()

    threshold = np.percentile(errors, threshold_pct)
    return pd.DataFrame(
        {"ae_error": errors, "ae_flag": errors > threshold}, index=features.index
    )


def detect_anomalies(returns: pd.DataFrame) -> pd.DataFrame:
    """Run both detectors and merge. `both_flag` marks high-conviction
    anomalies (both models agree)."""
    features = build_features(returns)
    out = isolation_forest_scores(features).join(autoencoder_scores(features))
    out["both_flag"] = out["if_flag"] & out["ae_flag"]
    return out


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    result = detect_anomalies(returns_wide())
    flagged = result[result["both_flag"]]
    print(f"{len(result)} days scored, {int(result['if_flag'].sum())} IF flags, "
          f"{int(result['ae_flag'].sum())} AE flags, {len(flagged)} agreed:")
    print(flagged.round(4))
