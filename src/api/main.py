"""Sentinel API — FastAPI service over the risk engine.

Endpoints:
  GET  /metrics       portfolio + per-ticker risk metrics
  GET  /forecast      GARCH(1,1) conditional-vol forecast + VaR term structure
  GET  /diagnostics   stationarity (ADF/KPSS), ACF/PACF, Ljung-Box ARCH-effect test
  POST /stress        run a named stress scenario
  GET  /anomalies     detected anomalous days
  GET  /fundamentals  EDGAR financial-statement ratios + DuPont
  GET  /forensic      distress / manipulation screens + Benford
  GET  /factors       Fama-French factor loadings + factor-adjusted alpha
  GET  /allocation    optimal portfolios (min-var / max-Sharpe / risk-parity)
  GET  /credit        Merton distance-to-default per name (structural credit)
  GET  /classify      supervised stress-day classifier (logistic + random forest)
  GET  /clusters      PCA statistical factors + KMeans peer clusters
  GET  /tactical      walk-forward backtest: the classifier gates the allocation
  GET  /analogs       k-nearest historical days to today + what followed
  GET  /candidates    semi-supervised near-miss anomaly days (label propagation)
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
    denoising: dict | None = Field(
        default=None,
        description="Marchenko-Pastur covariance-denoising check: realized "
        "out-of-sample vol of raw vs denoised min-variance weights")


class CreditResponse(BaseModel):
    distance_to_default: dict[str, dict] = Field(
        description="Per ticker (Merton 1974): distance-to-default, implied 1-year "
        "default probability, asset value/vol, leverage. Banks excluded.")


class ClassifyResponse(BaseModel):
    horizon_days: int = Field(description="Forward window the label looks over")
    threshold: float = Field(description="Drawdown that defines a 'stress day'")
    positive_rate: float = Field(description="Share of days labelled stress")
    n_train: int
    n_test: int
    models: dict[str, dict] = Field(
        description="Per model: out-of-sample ROC-AUC, time-series CV AUC, "
        "precision/recall, confusion matrix, ROC curve")
    feature_importance: dict[str, float] = Field(description="Random-forest importances")
    logistic_coef: dict[str, float] = Field(description="Standardized logistic coefficients")


class ClusterResponse(BaseModel):
    pca: dict = Field(description="Explained-variance ratios, cumulative, per-name loadings")
    kmeans: dict = Field(description="Best k, silhouette, per-ticker cluster labels")


class TacticalResponse(BaseModel):
    test_start: str
    test_days: int
    n_rebalances: int
    pct_defensive: float = Field(description="Share of days the gate held min-variance")
    gate: float
    cost_bps: float
    stats: dict[str, dict] = Field(
        description="Per book (tactical/aggressive/1-N): ann return, vol, "
        "Sharpe, max drawdown, Calmar — all out-of-sample")
    equity: dict[str, dict] = Field(description="Cumulative wealth per book, by date")
    rebalance_log: dict[str, dict] = Field(
        description="Per rebalance date: P(stress) and the regime chosen")


class AnalogResponse(BaseModel):
    query_date: str
    k: int
    summary: dict = Field(description="Forward-month distribution of the analogs")
    analogs: dict[str, dict] = Field(
        description="Per analog date: similarity distance, realised fwd 5d/21d return")


class CandidatesResponse(BaseModel):
    n_confirmed: int = Field(description="Both-detector anomaly seeds")
    n_candidates: int
    candidates: dict[str, dict] = Field(
        description="Near-miss days: propagated anomaly probability + which "
        "single detector (if any) fired")


class ForecastResponse(BaseModel):
    as_of: str
    horizon: int = Field(description="Trading days ahead the vol cone projects")
    var_horizons: list[int] = Field(description="Horizons (days) in the VaR term structure")
    series: dict[str, dict] = Field(
        description="Per ticker + equal-weight PORTFOLIO: GARCH(1,1) parameters "
        "(persistence, long-run vol), the annualized conditional-vol cone, and the "
        "95/99 forecast-VaR term structure (positive loss fractions)")
    ewma_vs_garch: dict = Field(
        description="Kupiec proportion-of-failures backtest comparing GARCH vs EWMA "
        "one-day-ahead VaR calibration on the portfolio")


class DiagnosticsResponse(BaseModel):
    as_of: str
    stationarity: dict = Field(
        description="ADF (H0: unit root) + KPSS (H0: stationary) on portfolio "
        "returns and on the log-price level — returns are stationary, the level is not")
    autocorrelation: dict = Field(
        description="ACF/PACF with confidence bands, plus Ljung-Box on returns "
        "(white-noise test) and on squared returns (ARCH-effect test that warrants GARCH)")


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


@lru_cache(maxsize=1)
def _forecast() -> dict:
    # one GARCH fit per name + portfolio + the backtest fit; a few seconds cold,
    # then cached for the process like the other heavy models.
    from src.models.volatility import forecast_summary
    return forecast_summary(_returns())


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast() -> ForecastResponse:
    try:
        return ForecastResponse(**_forecast())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Volatility forecast unavailable: {exc}")


@lru_cache(maxsize=1)
def _diagnostics() -> dict:
    from src.models.tsdiag import tsdiag_summary
    return tsdiag_summary(_returns())


@app.get("/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics() -> DiagnosticsResponse:
    try:
        return DiagnosticsResponse(**_diagnostics())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Diagnostics unavailable: {exc}")


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
        try:  # optional extra: the denoising sanity check may fail independently
            from src.models.unsupervised import denoising_backtest
            dn = denoising_backtest(_returns())
            denoising = {"realized_vol": dn["realized_vol"], **dn["info"]}
        except Exception:
            denoising = None
        return AllocationResponse(
            stats=json.loads(stats.to_json(orient="index")),
            weights=json.loads(weights.to_json()),
            denoising=denoising)
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


@app.get("/classify", response_model=ClassifyResponse)
def get_classify() -> ClassifyResponse:
    try:
        from src.models.classify import train_stress_classifier
        out = train_stress_classifier(_returns())
        return ClassifyResponse(**out)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Classifier unavailable: {exc}")


@app.get("/clusters", response_model=ClusterResponse)
def get_clusters() -> ClusterResponse:
    try:
        from src.models.unsupervised import cluster_universe, pca_factors
        r = _returns()
        p = pca_factors(r)
        return ClusterResponse(
            pca={"explained_variance_ratio": p["explained_variance_ratio"],
                 "cumulative_variance": p["cumulative_variance"],
                 "loadings": json.loads(p["loadings"].to_json(orient="index"))},
            kmeans=cluster_universe(r))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Clustering unavailable: {exc}")


@lru_cache(maxsize=1)
def _tactical() -> dict:
    # ~1-2 min cold (two walk-forward loops), then cached for the process
    from src.models.tactical import compare_overlays
    return compare_overlays(_returns())


@app.get("/tactical", response_model=TacticalResponse)
def get_tactical() -> TacticalResponse:
    try:
        out = _tactical()
        return TacticalResponse(
            test_start=out["test_start"], test_days=out["test_days"],
            n_rebalances=out["n_rebalances"], pct_defensive=out["pct_defensive"],
            gate=out["gate"], cost_bps=out["cost_bps"],
            stats=json.loads(out["stats"].to_json(orient="index")),
            equity=json.loads(out["equity"].round(4).to_json()),
            rebalance_log=json.loads(out["rebalance_log"].to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest unavailable: {exc}")


@app.get("/analogs", response_model=AnalogResponse)
def get_analogs() -> AnalogResponse:
    try:
        from src.models.analogs import analog_days
        out = analog_days(_returns())
        return AnalogResponse(
            query_date=out["query_date"], k=out["k"], summary=out["summary"],
            analogs=json.loads(out["analogs"].to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analogs unavailable: {exc}")


@app.get("/candidates", response_model=CandidatesResponse)
def get_candidates() -> CandidatesResponse:
    try:
        from src.models.semisup import propagate_labels
        out = propagate_labels(_returns(), _anomalies())
        cands = (out[out["candidate"]]
                 .sort_values("anomaly_prob", ascending=False)
                 [["anomaly_prob", "if_flag", "ae_flag"]])
        cands = cands.set_axis(cands.index.strftime("%Y-%m-%d"))  # not epoch-ms keys
        return CandidatesResponse(
            n_confirmed=int(out["both_flag"].sum()),
            n_candidates=int(len(cands)),
            candidates=json.loads(cands.to_json(orient="index")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Candidates unavailable: {exc}")


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
