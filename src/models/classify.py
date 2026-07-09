"""Supervised stress-day classifier — the course's W2 (logistic regression,
regularization, evaluation metrics) and W4 (ensemble) lens on Sentinel.

Every other ML model in Sentinel is UNSUPERVISED: IsolationForest and the
autoencoder learn "normal" and flag outliers with no labels. This module asks a
supervised question instead: given what markets look like today, can we predict
that a drawdown is coming?

We label each trading day of the equal-weight portfolio by its FORWARD outcome
(does the next `STRESS_FWD_DAYS` window draw down past `STRESS_DD_THRESHOLD`?),
build features that are all known by the close of that day, then train two
classifiers and score them honestly out-of-sample:

  • LogisticRegression with an L2 penalty      — W2: regression + regularization
  • RandomForestClassifier                     — W4: ensemble / bagging

The split is TEMPORAL (train on the earlier history, test on the later) so the
model never sees the future — a random split would leak tomorrow's regime into
training and flatter the AUC. XGBoost (W4 boosting) is picked up automatically
if installed but is never a hard dependency (keeps the light-footprint rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.risk import (ROLLING_WINDOW, drawdown_series, ewma_vol,
                             portfolio_returns, rolling_vol)

# why 21 / 0.05: a "stress day" is one where the portfolio falls at least 5%
# from here within the next trading month. ~1 month is short enough that today's
# conditions still carry signal; 5% is a real drawdown, not noise. Together they
# give a class balance (~15-25% positive) that trains without heavy re-weighting.
STRESS_FWD_DAYS = 21
STRESS_DD_THRESHOLD = 0.05

# why 0.70: fit on the first 70% of history, judge on the last 30% the model has
# never seen — the honest out-of-sample bar. Chronological, never shuffled.
TRAIN_FRACTION = 0.70

# why 5: TimeSeriesSplit CV folds — each validates on a block that comes strictly
# after its training block, the only cross-validation that respects time order.
CV_FOLDS = 5

RANDOM_STATE = 42


def _forward_stress_label(port: pd.Series) -> pd.Series:
    """Label each day 1 if the portfolio's cumulative return over the next
    STRESS_FWD_DAYS trading days troughs below -STRESS_DD_THRESHOLD, else 0.

    Uses only FUTURE returns (t+1..t+N), so it never overlaps the features
    (which are known by day t) — the label is the thing we're trying to predict.
    """
    # cumulative wealth multiple; forward path relative to today = w_{t+k}/w_t.
    wealth = (1 + port).cumprod()
    n = len(port)
    labels = np.zeros(n, dtype=int)
    vals = wealth.to_numpy()
    for t in range(n):
        end = min(t + STRESS_FWD_DAYS, n - 1)
        if end <= t:
            break
        # worst forward cumulative return over the horizon
        forward_min = vals[t + 1:end + 1].min() / vals[t] - 1
        labels[t] = int(forward_min <= -STRESS_DD_THRESHOLD)
    return pd.Series(labels, index=port.index, name="stress")


def build_dataset(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix X and forward-stress label y for the equal-weight portfolio.

    All six features are known by the close of day t; the label looks forward.
    The tail STRESS_FWD_DAYS rows (whose forward window runs off the data) are
    dropped so no half-observed label is ever trained on.
    """
    port = portfolio_returns(returns)
    feats = pd.DataFrame({
        # yesterday's shock, today's regime
        "ret_1d": port,
        "vol_21": rolling_vol(port, ROLLING_WINDOW),
        "ewma_vol": ewma_vol(port),
        # how far underwater we already are (drawdowns cluster)
        "drawdown": drawdown_series(port),
        # trailing one-month momentum (negative = already falling)
        "mom_21": port.rolling(ROLLING_WINDOW).sum(),
        # cross-sectional dispersion: wide spread across names precedes trouble
        "dispersion": returns.std(axis=1),
    })
    y = _forward_stress_label(port)
    data = feats.join(y).dropna()
    # drop the tail whose forward horizon is incomplete (label is all-zero there)
    data = data.iloc[:-STRESS_FWD_DAYS] if len(data) > STRESS_FWD_DAYS else data
    return data[feats.columns], data["stress"]


def _optional_boosters() -> dict:
    """W4 boosting, only if the user opted in by installing it — never required.

    XGBoost/LightGBM are heavier deps than the project's sklearn baseline, so we
    light them up opportunistically and stay silent when they're absent.
    """
    models = {}
    try:  # pragma: no cover - depends on an optional install
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE)
    except Exception:
        pass
    return models


def _evaluate(model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit on the training block, score out-of-sample + time-series CV."""
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # CV on the TRAINING block only, folds ordered in time (no leakage)
    cv = cross_val_score(model, X_tr, y_tr, cv=TimeSeriesSplit(CV_FOLDS),
                         scoring="roc_auc")
    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "cv_auc_mean": float(cv.mean()),
        "cv_auc_std": float(cv.std()),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "confusion": confusion_matrix(y_te, pred).tolist(),
        "roc_curve": {"fpr": [round(float(v), 4) for v in fpr],
                      "tpr": [round(float(v), 4) for v in tpr]},
    }


def train_stress_classifier(returns: pd.DataFrame) -> dict:
    """Train + evaluate the stress-day classifiers; return a dashboard/API payload.

    Logistic regression is standardized (scale matters for a penalized linear
    model); the random forest is scale-free so it takes raw features. Feature
    attribution comes from RF importances and the logistic coefficients — two
    independent reads on which market conditions actually precede drawdowns.
    """
    X, y = build_dataset(returns)
    split = int(len(X) * TRAIN_FRACTION)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # C controls the L2 strength (smaller C = stronger regularization); L2 is
    # sklearn's default penalty, so we set C rather than the deprecated `penalty` kwarg.
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000,
                           class_weight="balanced", random_state=RANDOM_STATE))
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)

    models = {"logistic": logit, "random_forest": forest, **_optional_boosters()}
    results = {name: _evaluate(m, X_tr, y_tr, X_te, y_te)
               for name, m in models.items()}

    # interpretability: RF importances + logistic |coef| (on standardized inputs)
    forest.fit(X_tr, y_tr)
    importances = dict(zip(X.columns, (round(float(v), 4)
                                       for v in forest.feature_importances_)))
    logit.fit(X_tr, y_tr)
    coefs = dict(zip(X.columns, (round(float(v), 4)
                                 for v in logit[-1].coef_[0])))

    return {
        "features": list(X.columns),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "positive_rate": round(float(y.mean()), 4),
        "horizon_days": STRESS_FWD_DAYS,
        "threshold": STRESS_DD_THRESHOLD,
        "models": results,
        "feature_importance": importances,
        "logistic_coef": coefs,
    }


if __name__ == "__main__":
    from src.warehouse.duck import ensure_loaded, returns_wide

    ensure_loaded()
    out = train_stress_classifier(returns_wide())
    print(f"Stress-day classifier - predict a >{out['threshold']:.0%} drawdown "
          f"within {out['horizon_days']} days")
    print(f"  train={out['n_train']}  test={out['n_test']}  "
          f"positive_rate={out['positive_rate']:.1%}\n")
    for name, r in out["models"].items():
        print(f"  {name:14} test ROC-AUC {r['roc_auc']:.3f}  "
              f"(CV {r['cv_auc_mean']:.3f}+/-{r['cv_auc_std']:.3f})  "
              f"precision {r['precision']:.2f}  recall {r['recall']:.2f}")
    print("\n  RF feature importance:")
    for f, v in sorted(out["feature_importance"].items(),
                       key=lambda kv: -kv[1]):
        print(f"    {f:12} {v:.3f}")
