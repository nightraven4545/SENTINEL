"""Sentinel API — FastAPI service over the risk engine.

Endpoints:
  GET  /metrics       portfolio + per-ticker risk metrics
  POST /stress        run a named stress scenario
  GET  /anomalies     detected anomalous days
  GET  /fundamentals  EDGAR financial-statement ratios + DuPont
  GET  /forensic      distress / manipulation screens + Benford
  GET  /factors       Fama-French factor loadings + factor-adjusted alpha
  GET  /allocation    optimal portfolios (min-var / max-Sharpe / risk-parity)
  GET  /credit        Merton distance-to-default per name (structural credit)
  POST /ask           the agent answers a free-text question (tool-use)
  GET  /memo          generate the risk memo (LLM or template fallback)
  GET  /health        liveness probe (used by Docker healthcheck)

Heavy computations (warehouse read, anomaly-model training) are cached per
process — the models are deterministic over a fixed dataset, so recomputing
per request would be pure waste.
"""
from __future__ import annotations

import datetime as dt
import json
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import memo as agent
from src.models import risk, stress
from src.models.anomaly import detect_anomalies


# ---------------------------------------------------------------- lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap the warehouse on first run so a fresh container serves
    real data after a single `docker run` (fetch needs outbound network)."""
    from src.warehouse.duck import ensure_loaded

    ensure_loaded()
    yield


app = FastAPI(
    title="Sentinel API",
    version="0.1.0",
    description="Agentic quant risk & anomaly engine — "
                "*a junior quant risk analyst, automated.*",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def _returns() -> pd.DataFrame:
    from src.warehouse.duck import returns_wide
    return returns_wide()


@lru_cache(maxsize=1)
def _anomalies() -> pd.DataFrame:
    return detect_anomalies(_returns())


# ---------------------------------------------------------------- schemas

class RiskMetrics(BaseModel):
    ann_return: float
    ann_vol: float
    sharpe: float = Field(description="Annualized excess return / total vol")
    sortino: float = Field(description="Annualized excess return / downside deviation")
    var_95: float = Field(description="Historical daily VaR 95% (positive loss fraction)")
    cf_var_95: float = Field(description="Cornish-Fisher modified VaR 95% (skew/kurtosis-adjusted)")
    es_95: float = Field(description="Expected Shortfall 95%: mean loss beyond VaR")
    ewma_var_95: float = Field(description="Regime-aware VaR 95% from the latest EWMA (RiskMetrics) vol")
    var_99: float
    max_drawdown: float = Field(description="Worst peak-to-trough loss (negative)")
    skew: float
    ex_kurtosis: float


class MetricsResponse(BaseModel):
    as_of: dt.date
    metrics: dict[str, RiskMetrics] = Field(
        description="Per ticker, plus a 'PORTFOLIO' equal-weight row")


class StressMetrics(BaseModel):
    ann_return: float
    ann_vol: float
    var_95: float
    max_drawdown: float


class StressRequest(BaseModel):
    scenario: str = Field(examples=list(stress.SCENARIOS))


class StressResponse(BaseModel):
    scenario: str
    description: str
    baseline: StressMetrics
    stressed: StressMetrics


class AnomalyDay(BaseModel):
    date: dt.date
    if_score: float
    ae_error: float
    if_flag: bool
    ae_flag: bool
    both_flag: bool = Field(description="Both models agree — high conviction")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


class MemoResponse(BaseModel):
    markdown: str
    generated_with_llm: bool


class FundamentalsResponse(BaseModel):
    ratios: dict[str, dict] = Field(description="Per ticker: liquidity/solvency/"
                                    "profitability/efficiency ratios")
    dupont: dict[str, dict] = Field(description="Per ticker: 3-step DuPont ROE")


class ForensicResponse(BaseModel):
    scores: dict[str, dict] = Field(description="Per ticker: Altman Z, Piotroski F, "
                                    "Beneish M, accruals")
    benford: dict = Field(description="Benford first-digit MAD + verdict")


class FactorResponse(BaseModel):
    portfolio: dict = Field(description="Factor-adjusted alpha, R², betas, t-stats")
    loadings: dict[str, dict] = Field(description="Per-ticker factor betas + alpha")


class AllocationResponse(BaseModel):
    stats: dict[str, dict] = Field(description="return/vol/Sharpe per portfolio")
    weights: dict[str, dict] = Field(description="Per portfolio: ticker weights")


class CreditResponse(BaseModel):
    distance_to_default: dict[str, dict] = Field(
        description="Per ticker (Merton 1974): distance-to-default, implied 1-year "
        "default probability, asset value/vol, leverage. Banks excluded.")


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    r = _returns()
    table = risk.summary(r).round(6)
    return MetricsResponse(as_of=r.index.max().date(),
                           metrics=table.to_dict(orient="index"))


@app.post("/stress", response_model=StressResponse)
def run_stress(req: StressRequest) -> StressResponse:
    scenario = stress.SCENARIOS.get(req.scenario)
    if scenario is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{req.scenario}'. "
                   f"Valid scenarios: {list(stress.SCENARIOS)}")
    r = _returns()
    return StressResponse(
        scenario=scenario.name,
        description=scenario.description,
        baseline=stress.portfolio_metrics(r),
        stressed=stress.run_scenario(r, scenario.name),
    )


@app.get("/anomalies", response_model=list[AnomalyDay])
def get_anomalies(flagged_only: bool = True) -> list[AnomalyDay]:
    df = _anomalies()
    if flagged_only:
        df = df[df["if_flag"] | df["ae_flag"]]
    return [AnomalyDay(date=idx.date(), **row)
            for idx, row in df.round(6).iterrows()]


@app.get("/fundamentals", response_model=FundamentalsResponse)
def get_fundamentals() -> FundamentalsResponse:
    try:
        from src.ingest.edgar import latest_fundamentals
        from src.models.fundamentals import dupont, ratios
        w = latest_fundamentals()
        return FundamentalsResponse(
            ratios=json.loads(ratios(w).to_json(orient="index")),
            dupont=json.loads(dupont(w).to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fundamentals unavailable: {exc}")


@app.get("/forensic", response_model=ForensicResponse)
def get_forensic() -> ForensicResponse:
    try:
        from src.ingest.edgar import fetch_fundamentals, latest_fundamentals
        from src.ingest.market import fetch_prices
        from src.models import forensic as fx
        long = fetch_fundamentals()
        close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
        shares = latest_fundamentals(long).get("shares")
        mcap = (shares * close).dropna() if shares is not None else None
        return ForensicResponse(
            scores=json.loads(fx.forensic_summary(long, mcap).to_json(orient="index")),
            benford=fx.benford_mad(long))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Forensic scores unavailable: {exc}")


@app.get("/factors", response_model=FactorResponse)
def get_factors() -> FactorResponse:
    try:
        from src.ingest.factors import fetch_factors
        from src.models.factors import factor_loadings, portfolio_factor_model
        f = fetch_factors()
        m = portfolio_factor_model(_returns(), f)
        return FactorResponse(
            portfolio={k: m[k] for k in ("alpha_ann", "alpha_t", "r2", "betas", "tstats")},
            loadings=json.loads(factor_loadings(_returns(), f).to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Factor model unavailable: {exc}")


@app.get("/allocation", response_model=AllocationResponse)
def get_allocation() -> AllocationResponse:
    try:
        from src.models.optimize import compare_portfolios
        stats, weights = compare_portfolios(_returns())
        return AllocationResponse(
            stats=json.loads(stats.to_json(orient="index")),
            weights=json.loads(weights.to_json()))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Optimization unavailable: {exc}")


@app.get("/credit", response_model=CreditResponse)
def get_credit() -> CreditResponse:
    try:
        from src.ingest.edgar import latest_fundamentals
        from src.ingest.market import fetch_prices
        from src.models.credit import default_point, distance_to_default
        w = latest_fundamentals()
        close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
        shares = w.get("shares")
        if shares is None:
            raise ValueError("shares outstanding unavailable")
        mcap = (shares * close).dropna()
        dtd = distance_to_default(_returns(), mcap, default_point(w))
        return CreditResponse(
            distance_to_default=json.loads(dtd.to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Credit model unavailable: {exc}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return AskResponse(answer=agent.ask(req.question))
    except Exception as exc:  # LLM/provider failure -> upstream error, not 500
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}")


@app.get("/memo", response_model=MemoResponse)
def memo() -> MemoResponse:
    try:
        markdown = agent.write_memo()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Memo generation failed: {exc}")
    return MemoResponse(markdown=markdown,
                        generated_with_llm=agent._client() is not None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
