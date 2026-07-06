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
                        "cf_var_95", "es_95", "var_99", "max_drawdown", "skew",
                        "ex_kurtosis"}
    assert row["var_95"] > 0 and row["max_drawdown"] <= 0
    assert row["es_95"] >= row["var_95"]  # tail mean is at least the threshold


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
