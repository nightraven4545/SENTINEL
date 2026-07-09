# Course mapping — IITG.ai "ML.AI" summer course → Sentinel

This maps every week of the [IITG.ai](https://iitgai.in) Summer Course on Data
Science & Machine Learning onto the Sentinel module that puts it to work on a
real quant-finance problem. The course teaches the classical ML toolkit end to
end (Python → supervised → unsupervised → ensembles → neural networks); Sentinel
is where those techniques earn their keep.

## Week-by-week

| Week | Course topics | Where it lives in Sentinel |
|---|---|---|
| **W1** — Python, NumPy, Pandas, Matplotlib, EDA | data wrangling, plotting, exploratory analysis | The whole data layer: [`src/ingest`](../src/ingest) loaders, the [`src/warehouse`](../src/warehouse) DuckDB store, and the narrative EDA in [`notebooks/`](../notebooks). Every model consumes a tidy wide returns frame. |
| **W2** — Linear & **Logistic Regression**, Regularization, Evaluation Metrics, Sklearn | supervised learning, train/test discipline, ROC/precision/recall | **Linear:** the hand-rolled OLS factor regression in [`src/models/factors.py`](../src/models/factors.py) and CAPM in [`src/models/risk.py`](../src/models/risk.py). **Logistic + regularization + metrics (new):** [`src/models/classify.py`](../src/models/classify.py) — an L2-penalized logistic stress-day classifier scored out-of-sample with ROC-AUC, precision, recall and a confusion matrix. |
| **W3** — SVM, KMeans, KNN, **PCA**, dimensionality reduction, clustering, anomaly detection | unsupervised structure | **Anomaly detection:** IsolationForest + autoencoder in [`src/models/anomaly.py`](../src/models/anomaly.py). **PCA + KMeans (new):** [`src/models/unsupervised.py`](../src/models/unsupervised.py) — PCA extracts the statistical risk factors (PC1 ≈ the market), KMeans clusters the names by correlation profile and recovers the same sector blocks the [correlation network](../src/models/graph.py) finds. |
| **W4** — **Ensemble / boosting** (XGBoost, CatBoost, LightGBM), semi-supervised, Kaggle | bagging, boosting, model competitions | **Ensemble:** the `RandomForestClassifier` in [`src/models/classify.py`](../src/models/classify.py) runs alongside the logistic model, with feature importances. XGBoost is picked up automatically *if installed* (`pip install xgboost`) but is never a hard dependency. |
| **W5** — Neural networks & deep learning (gradient descent, activations, backprop) | feed-forward nets, PyTorch | The **PyTorch autoencoder** in [`src/models/anomaly.py`](../src/models/anomaly.py): an encoder/decoder trained by backprop to reconstruct a "normal" market day; days it reconstructs poorly are anomalies. Gradient descent, activations and a train loop, applied not classified-cats-and-dogs but market risk. |
| **Additional Materials** | *locked / not yet released* | Revisit when the course unlocks it. |

## What was built specifically for this mapping

Three sklearn-only modules (no new dependency) fill the genuine gaps between the
syllabus and what Sentinel already had:

- **[`classify.py`](../src/models/classify.py)** — supervised stress-day classifier.
  Labels each day by its *forward* 21-day drawdown (a >5% drop ahead = "stress"),
  builds six features known by that day's close (rolling & EWMA vol, drawdown,
  momentum, dispersion, last return), and trains Logistic Regression + Random
  Forest with a **chronological** train/test split and time-series cross-validation
  — the honest way to evaluate a time series (a random split leaks the future).
  Out-of-sample logistic ROC-AUC ≈ 0.73; volatility is the dominant predictor.
- **[`unsupervised.py`](../src/models/unsupervised.py)** — PCA + KMeans. PC1 explains
  ~41% of variance as a single common "market" factor; KMeans on correlation
  profiles recovers tech / defensives / cyclicals-&-energy, matching the network
  communities exactly.

Both are surfaced in the API (`GET /classify`, `GET /clusters`), the Streamlit
dashboard (the **ML Models** tab), and pinned by oracle tests in
[`tests/test_ml.py`](../tests/test_ml.py).

## Deliberately not forced in

Naïve Bayes, SVM and KNN (W2/W3) are one-line sklearn swaps but weak fits for
this data (continuous, autocorrelated financial series with heavy class
imbalance), so they're noted here rather than bolted on for show. XGBoost /
CatBoost / LightGBM (W4) are heavier dependencies than the project's sklearn
baseline, so they stay opt-in. The point of the mapping is to use each technique
where it genuinely belongs — not to check every box.
