"""API tests over synthetic data — no network, no warehouse, no LLM."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api


@pytest.fixture(autouse=True)
def synthetic_data(monkeypatch):
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 3)), columns=["A", "B", "C"],
                      index=pd.bdate_range("2024-01-01", periods=300))
    monkeypatch.setattr(api, "_returns", lambda: df)
    api._anomalies.cache_clear()
    api._tactical.cache_clear()
    api._forecast.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def client():
    # plain TestClient skips lifespan -> no warehouse bootstrap in tests
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_includes_portfolio_row(client):
    body = client.get("/metrics").json()
    assert "PORTFOLIO" in body["metrics"]
    row = body["metrics"]["PORTFOLIO"]
    assert set(row) == {"ann_return", "ann_vol", "sharpe", "sortino", "var_95",
                        "cf_var_95", "es_95", "ewma_var_95", "var_99",
                        "max_drawdown", "skew", "ex_kurtosis"}
    assert row["var_95"] > 0 and row["max_drawdown"] <= 0
    assert row["es_95"] >= row["var_95"]  # tail mean is at least the threshold
    assert row["ewma_var_95"] > 0  # today's regime-aware VaR is a positive loss


def test_forecast_endpoint(client):
    body = client.get("/forecast").json()
    assert body["horizon"] == 21
    assert body["var_horizons"] == [1, 5, 10, 21]
    assert "PORTFOLIO" in body["series"]
    p = body["series"]["PORTFOLIO"]
    assert len(p["vol_path"]) == 21
    # forecast-VaR term structure: positive and monotone-increasing in horizon
    vals = [p["var_95"][k] for k in sorted(p["var_95"], key=int)]
    assert all(v > 0 for v in vals) and vals == sorted(vals)
    assert {"garch", "ewma"} <= set(body["ewma_vs_garch"])


def test_stress_valid_scenario(client):
    body = client.post("/stress", json={"scenario": "market_crash_2008_style"}).json()
    assert body["stressed"]["var_95"] > body["baseline"]["var_95"]
    assert "crash" in body["description"].lower()


def test_stress_unknown_scenario_404_lists_valid(client):
    resp = client.post("/stress", json={"scenario": "zombie_apocalypse"})
    assert resp.status_code == 404
    assert "rate_shock" in resp.json()["detail"]


def test_stress_missing_body_422(client):
    assert client.post("/stress", json={}).status_code == 422


def test_anomalies_flagged_only(client):
    days = client.get("/anomalies").json()
    assert all(d["if_flag"] or d["ae_flag"] for d in days)
    assert {"date", "if_score", "ae_error", "both_flag"} <= set(days[0])


def test_allocation_endpoint(client, monkeypatch):
    # patch the optimiser (lazily imported in the endpoint) -> hermetic wrapper test
    import src.models.optimize as opt
    stats = pd.DataFrame({"Equal-weight": {"return": 0.1, "vol": 0.15, "sharpe": 0.4},
                          "Max-Sharpe": {"return": 0.2, "vol": 0.18, "sharpe": 0.9}}).T
    weights = pd.DataFrame({"Equal-weight": {"A": 0.5, "B": 0.5},
                            "Max-Sharpe": {"A": 0.7, "B": 0.3}})
    monkeypatch.setattr(opt, "compare_portfolios", lambda r: (stats, weights))
    body = client.get("/allocation").json()
    assert set(body["stats"]) == {"Equal-weight", "Max-Sharpe"}
    assert body["weights"]["Max-Sharpe"]["A"] == pytest.approx(0.7)


def test_factors_endpoint(client, monkeypatch):
    # patch the factor library (lazily imported) with synthetic factors -> hermetic
    import src.ingest.factors as ff
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(5)
    fac = pd.DataFrame({c: rng.normal(0, 0.01, 300) for c in ff.FACTORS}, index=idx)
    fac["rf"] = 0.0001
    monkeypatch.setattr(ff, "fetch_factors", lambda *a, **k: fac)
    body = client.get("/factors").json()
    assert {"alpha_ann", "r2", "betas", "tstats"} <= set(body["portfolio"])
    assert "mkt_rf" in body["portfolio"]["betas"]


def test_credit_endpoint(client, monkeypatch):
    # patch the Merton solver (lazily imported) + trivial valid mcap inputs
    import src.ingest.edgar as edgar
    import src.ingest.market as market
    import src.models.credit as credit
    w = pd.DataFrame({"shares": {"A": 10.0, "B": 20.0},
                      "current_liabilities": {"A": 5.0, "B": 8.0},
                      "long_term_debt": {"A": 2.0, "B": 3.0}})
    monkeypatch.setattr(edgar, "latest_fundamentals", lambda *a, **k: w)
    prices = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                           "ticker": ["A", "B"], "close": [100.0, 50.0]})
    monkeypatch.setattr(market, "fetch_prices", lambda *a, **k: prices)
    dtd = pd.DataFrame({"distance_to_default": {"A": 5.0, "B": 3.0},
                        "default_prob": {"A": 1e-7, "B": 1e-3},
                        "leverage": {"A": 0.1, "B": 0.2}})
    monkeypatch.setattr(credit, "distance_to_default", lambda *a, **k: dtd)
    body = client.get("/credit").json()
    assert set(body["distance_to_default"]) == {"A", "B"}
    assert body["distance_to_default"]["B"]["distance_to_default"] == pytest.approx(3.0)


def test_classify_endpoint(client, monkeypatch):
    # patch the classifier (lazily imported in the endpoint) -> hermetic wrapper test
    import src.models.classify as clf
    canned = {"features": ["vol_21"], "horizon_days": 21, "threshold": 0.05,
              "positive_rate": 0.15, "n_train": 100, "n_test": 40,
              "models": {"logistic": {"roc_auc": 0.71, "cv_auc_mean": 0.55,
                                      "precision": 0.3, "recall": 0.6,
                                      "confusion": [[30, 5], [3, 2]],
                                      "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}}},
              "feature_importance": {"vol_21": 0.4}, "logistic_coef": {"vol_21": 0.3}}
    monkeypatch.setattr(clf, "train_stress_classifier", lambda r: canned)
    body = client.get("/classify").json()
    assert body["horizon_days"] == 21 and body["positive_rate"] == pytest.approx(0.15)
    assert body["models"]["logistic"]["roc_auc"] == pytest.approx(0.71)


def test_clusters_endpoint(client, monkeypatch):
    # patch PCA + KMeans (lazily imported) -> hermetic wrapper test
    import src.models.unsupervised as un
    pca = {"explained_variance_ratio": [0.4, 0.1], "cumulative_variance": [0.4, 0.5],
           "loadings": pd.DataFrame({"PC1": {"A": 0.3, "B": 0.4}}),
           "n_obs": 100, "n_assets": 2}
    km = {"k": 2, "silhouette": 0.5, "labels": {"A": 0, "B": 1},
          "silhouette_by_k": {2: 0.5}}
    monkeypatch.setattr(un, "pca_factors", lambda r: pca)
    monkeypatch.setattr(un, "cluster_universe", lambda r: km)
    body = client.get("/clusters").json()
    assert body["kmeans"]["k"] == 2 and body["kmeans"]["labels"]["B"] == 1
    assert body["pca"]["loadings"]["A"]["PC1"] == pytest.approx(0.3)


def test_tactical_endpoint(client, monkeypatch):
    # patch the walk-forward engine (lazily imported) -> hermetic wrapper test
    import src.models.tactical as tac
    idx = pd.bdate_range("2025-01-01", periods=3)
    daily = pd.DataFrame({"tactical_max_sharpe": [0.01, -0.02, 0.005],
                          "always_max_sharpe": [0.02, -0.03, 0.004],
                          "equal_weight": [0.01, -0.01, 0.002]}, index=idx)
    canned = {"test_start": "2025-01-01", "test_days": 3, "n_rebalances": 1,
              "pct_defensive": 0.4, "gate": 0.5, "cost_bps": 5,
              "stats": pd.DataFrame({"sharpe": {"tactical_max_sharpe": 1.0}}),
              "equity": (1 + daily).cumprod(),
              "rebalance_log": pd.DataFrame(
                  {"p_stress": [0.6], "defensive": [True]},
                  index=pd.Index(idx[:1], name="date"))}
    monkeypatch.setattr(tac, "compare_overlays", lambda r: canned)
    body = client.get("/tactical").json()
    assert body["pct_defensive"] == pytest.approx(0.4)
    assert body["stats"]["tactical_max_sharpe"]["sharpe"] == pytest.approx(1.0)
    assert len(body["equity"]["tactical_max_sharpe"]) == 3


def test_analogs_endpoint(client, monkeypatch):
    import src.models.analogs as an
    canned = {"query_date": "2025-06-30", "k": 2,
              "summary": {"median_fwd_21d": 0.01, "pct_negative": 0.5},
              "analogs": pd.DataFrame(
                  {"distance": [0.1, 0.2], "fwd_5d": [0.01, -0.02],
                   "fwd_21d": [0.03, -0.04]},
                  index=pd.Index(pd.to_datetime(["2020-01-02", "2021-05-04"]),
                                 name="date")),
              "paths": {}}
    monkeypatch.setattr(an, "analog_days", lambda r: canned)
    body = client.get("/analogs").json()
    assert body["k"] == 2 and body["summary"]["median_fwd_21d"] == pytest.approx(0.01)
    assert len(body["analogs"]) == 2


def test_candidates_endpoint(client, monkeypatch):
    import src.models.semisup as ss
    idx = pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01"])
    out = pd.DataFrame({"anomaly_prob": [0.9, 0.2, 0.95],
                        "if_flag": [True, False, False],
                        "ae_flag": [False, False, False],
                        "both_flag": [False, False, False],
                        "candidate": [True, False, True]}, index=idx)
    monkeypatch.setattr(ss, "propagate_labels", lambda r, a: out)
    body = client.get("/candidates").json()
    assert body["n_candidates"] == 2
    # ranked: the 0.95 day first, keyed by readable date
    assert list(body["candidates"])[0] == "2024-04-01"
    probs = [c["anomaly_prob"] for c in body["candidates"].values()]
    assert probs == sorted(probs, reverse=True)


def test_ask_without_llm_returns_answer(client):
    body = client.post("/ask", json={"question": "What is the VaR?"}).json()
    assert "ANTHROPIC_API_KEY" in body["answer"]


def test_ask_empty_question_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_memo_template_mode(client, monkeypatch):
    # memo's gather_context reads its own cached returns — patch it too
    from src.agent import memo as agent
    monkeypatch.setattr(agent, "_returns", api._returns)
    body = client.get("/memo").json()
    assert body["generated_with_llm"] is False
    assert body["markdown"].startswith("# Sentinel Risk Memo")
