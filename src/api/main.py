"""Sentinel API — FastAPI service over the risk engine.

Endpoints:
  GET  /metrics    portfolio + per-ticker risk metrics
  POST /stress     run a named stress scenario
  GET  /anomalies  detected anomalous days
  POST /ask        the agent answers a free-text question (tool-use)
  GET  /memo       generate the risk memo (LLM or template fallback)
  GET  /health     liveness probe (used by Docker healthcheck)

Heavy computations (warehouse read, anomaly-model training) are cached per
process — the models are deterministic over a fixed dataset, so recomputing
per request would be pure waste.
"""
from __future__ import annotations

import datetime as dt
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
    var_95: float = Field(description="Historical daily VaR 95% (positive loss fraction)")
    var_99: float
    max_drawdown: float = Field(description="Worst peak-to-trough loss (negative)")


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
