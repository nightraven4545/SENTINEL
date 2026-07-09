"""The GenAI risk analyst: writes a structured Risk Memo and answers
free-text questions by calling Sentinel's own model functions (tool use).

Two entry points:
- write_memo()  -> markdown memo (LLM if configured, honest template if not)
- ask(question) -> agentic loop: Claude calls our risk/stress/anomaly/graph
                   functions as tools until it can answer.

Provider: Anthropic API, configured via .env (ANTHROPIC_API_KEY,
SENTINEL_LLM_MODEL). With no key the app still runs — memo falls back to a
deterministic template and ask() says so instead of failing.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv

from src.models import risk, stress
from src.models.anomaly import detect_anomalies
from src.models.graph import graph_payload

load_dotenv()

DEFAULT_MODEL = "claude-opus-4-8"
MEMO_MAX_TOKENS = 8000
ASK_MAX_TOKENS = 4000
MAX_TOOL_TURNS = 8  # hard stop so a confused loop can't run forever

SYSTEM = (
    "You are Sentinel, a junior quantitative risk analyst. You analyse a "
    "10-name cross-sector equity portfolio using tools that call the firm's "
    "own risk engine. Be precise with numbers (quote them from tool results, "
    "never invent them), state assumptions, and write in clear analyst prose. "
    "Daily VaR is reported as a positive loss fraction; drawdowns are negative."
)


# ---------------------------------------------------------------- data layer

@lru_cache(maxsize=1)
def _returns() -> pd.DataFrame:
    """Load returns from the warehouse once per process."""
    from src.warehouse.duck import returns_wide
    return returns_wide()


def gather_context() -> dict:
    """Everything the memo writer needs, computed eagerly as plain data."""
    r = _returns()
    anoms = detect_anomalies(r)
    agreed = anoms[anoms["both_flag"]]
    payload = graph_payload(r)
    port = risk.portfolio_returns(r)
    try:  # CAPM vs SPY — optional: memo still works if the benchmark is missing
        from src.ingest.market import fetch_benchmark
        capm_vs_spy = {k: round(v, 4) for k, v in
                       risk.capm(port, fetch_benchmark()).items()}
    except Exception:
        capm_vs_spy = None
    return {
        "as_of": str(r.index.max().date()),
        "tickers": list(r.columns),
        "risk_summary": risk.summary(r).round(4).to_dict(orient="index"),
        "capm_vs_spy": capm_vs_spy,
        "risk_contributions": risk.risk_contributions(r).round(4)
                                  .to_dict(orient="index"),
        "var_backtest_kupiec": risk.kupiec_test(port),
        "anomalies": {
            "if_flags": int(anoms["if_flag"].sum()),
            "ae_flags": int(anoms["ae_flag"].sum()),
            "agreed_dates": [str(d.date()) for d in agreed.index],
        },
        "graph": payload,
        "stress": stress.compare(r).round(4).to_dict(orient="index"),
        "scenario_descriptions": {n: s.description for n, s in stress.SCENARIOS.items()},
        # The CA + CFA depth lenses — each optional so the memo still writes if
        # EDGAR / the factor library are unreachable (e.g. offline).
        "factor_model": _factor_context(r),
        "fundamentals": _fundamentals_context(),
        "forensic": _forensic_context(),
        "credit": _credit_context(r),
        "allocation": _allocation_context(r),
    }


def _json_records(df) -> dict:
    """DataFrame -> plain dict with NaN/None as null (pandas to_json handles the
    NaN and bool coercion that json.dumps chokes on)."""
    return json.loads(df.to_json(orient="index"))


def _factor_context(r: pd.DataFrame) -> dict | None:
    """Fama-French factor loadings + factor-adjusted alpha for the portfolio."""
    try:
        from src.ingest.factors import fetch_factors
        from src.models.factors import portfolio_factor_model
        m = portfolio_factor_model(r, fetch_factors())
        return {"alpha_ann": round(m["alpha_ann"], 4), "alpha_t": round(m["alpha_t"], 2),
                "r2": round(m["r2"], 3),
                "betas": {k: round(v, 3) for k, v in m["betas"].items()},
                "tstats": {k: round(v, 2) for k, v in m["tstats"].items()}}
    except Exception:
        return None


def _latest_market_cap():
    """Market value of equity per ticker (shares × latest close) for Altman."""
    from src.ingest.edgar import latest_fundamentals
    from src.ingest.market import fetch_prices
    close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
    shares = latest_fundamentals().get("shares")
    return (shares * close).dropna() if shares is not None else None


def _fundamentals_context() -> dict | None:
    """Financial-statement ratios + DuPont from EDGAR (CA lens)."""
    try:
        from src.ingest.edgar import latest_fundamentals
        from src.models.fundamentals import ratios
        rat = ratios(latest_fundamentals())
        return {"median_roe": round(float(rat["roe"].median()), 4),
                "median_net_margin": round(float(rat["net_margin"].median()), 4),
                "median_debt_to_equity": round(float(rat["debt_to_equity"].median()), 4),
                "ratios": _json_records(rat.round(4))}
    except Exception:
        return None


def _forensic_context() -> dict | None:
    """Distress / manipulation screens (Altman, Piotroski, Beneish, Benford)."""
    try:
        from src.ingest.edgar import fetch_fundamentals
        from src.models import forensic as fx
        long = fetch_fundamentals()
        summ = fx.forensic_summary(long, _latest_market_cap())
        return {
            "altman_distress": [t for t, z in summ["altman_z"].items()
                                if pd.notna(z) and z < 1.81],
            "piotroski_strong": [t for t, f in summ["piotroski_f"].items() if f >= 7],
            "beneish_flags": [t for t, m in summ["m_flag"].items() if m is True],
            "benford": fx.benford_mad(long),
            "scores": _json_records(summ),
        }
    except Exception:
        return None


def _credit_context(r: pd.DataFrame) -> dict | None:
    """Merton structural distance-to-default per name — the market's read on the
    same distress the Altman screen scores off the balance sheet."""
    try:
        from src.models.credit import default_point, distance_to_default
        mcap = _latest_market_cap()
        if mcap is None:
            return None
        dtd = distance_to_default(r, mcap, default_point())
        valid = dtd["distance_to_default"].dropna()
        if valid.empty:
            return None
        return {
            "thinnest_name": valid.idxmin(),
            "thinnest_dd": round(float(valid.min()), 2),
            "median_dd": round(float(valid.median()), 2),
            "distance_to_default": _json_records(
                dtd[["distance_to_default", "default_prob", "leverage"]].round(4)),
        }
    except Exception:
        return None


def _allocation_context(r: pd.DataFrame) -> dict | None:
    """Optimal portfolios vs the 1/N book (CFA lens)."""
    try:
        from src.models.optimize import compare_portfolios
        stats, weights = compare_portfolios(r)
        return {"stats": _json_records(stats.round(4)),
                "weights": json.loads(weights.round(4).to_json())}
    except Exception:
        return None


# ---------------------------------------------------------------- tools

def _tool_risk_summary() -> str:
    return risk.summary(_returns()).round(4).to_json(orient="index")


def _tool_anomalies(limit: int = 20) -> str:
    anoms = detect_anomalies(_returns())
    agreed = anoms[anoms["both_flag"]].tail(limit)
    agreed = agreed.set_axis(agreed.index.strftime("%Y-%m-%d"))
    return agreed.round(4).to_json(orient="index")


def _tool_graph() -> str:
    return json.dumps(graph_payload(_returns()))


def _tool_stress(scenario: str) -> str:
    return json.dumps(stress.run_scenario(_returns(), scenario))


def _tool_compare_scenarios() -> str:
    return stress.compare(_returns()).round(4).to_json(orient="index")


def _tool_fundamentals() -> str:
    from src.ingest.edgar import latest_fundamentals
    from src.models.fundamentals import dupont, ratios
    w = latest_fundamentals()
    return json.dumps({"ratios": _json_records(ratios(w)),
                       "dupont": _json_records(dupont(w))})


def _tool_forensic() -> str:
    from src.ingest.edgar import fetch_fundamentals
    from src.models import forensic as fx
    long = fetch_fundamentals()
    return json.dumps({"scores": _json_records(fx.forensic_summary(long, _latest_market_cap())),
                       "benford": fx.benford_mad(long)})


def _tool_factor_model() -> str:
    from src.ingest.factors import fetch_factors
    from src.models.factors import factor_loadings, portfolio_factor_model
    f = fetch_factors()
    m = portfolio_factor_model(_returns(), f)
    return json.dumps({
        "portfolio": {k: m[k] for k in ("alpha_ann", "alpha_t", "r2", "betas", "tstats")},
        "loadings": _json_records(factor_loadings(_returns(), f))})


def _tool_optimization() -> str:
    from src.models.optimize import compare_portfolios
    stats, weights = compare_portfolios(_returns())
    return json.dumps({"stats": _json_records(stats),
                       "weights": json.loads(weights.to_json())})


def _tool_credit() -> str:
    from src.models.credit import default_point, distance_to_default
    dtd = distance_to_default(_returns(), _latest_market_cap(), default_point())
    return dtd.round(4).to_json(orient="index")


def _tool_analogs() -> str:
    from src.models.analogs import analog_days
    out = analog_days(_returns())
    a = out["analogs"].set_axis(out["analogs"].index.strftime("%Y-%m-%d"))
    return json.dumps({"query_date": out["query_date"], "k": out["k"],
                       "summary": out["summary"],
                       "analogs": json.loads(a.to_json(orient="index"))})


def _tool_tactical() -> str:
    from src.models.tactical import compare_overlays
    out = compare_overlays(_returns())
    return json.dumps({"stats": _json_records(out["stats"]),
                       "pct_defensive": out["pct_defensive"],
                       "test_start": out["test_start"],
                       "test_days": out["test_days"], "gate": out["gate"]})


TOOL_FUNCS = {
    "get_risk_summary": lambda inp: _tool_risk_summary(),
    "get_anomalies": lambda inp: _tool_anomalies(int(inp.get("limit", 20))),
    "get_graph_findings": lambda inp: _tool_graph(),
    "run_stress_scenario": lambda inp: _tool_stress(inp["scenario"]),
    "compare_all_scenarios": lambda inp: _tool_compare_scenarios(),
    "get_fundamentals": lambda inp: _tool_fundamentals(),
    "get_forensic_scores": lambda inp: _tool_forensic(),
    "get_factor_model": lambda inp: _tool_factor_model(),
    "get_optimal_portfolios": lambda inp: _tool_optimization(),
    "get_distance_to_default": lambda inp: _tool_credit(),
    "get_analog_days": lambda inp: _tool_analogs(),
    "get_tactical_backtest": lambda inp: _tool_tactical(),
}

TOOLS = [
    {
        "name": "get_risk_summary",
        "description": "Per-ticker and equal-weight-portfolio risk metrics: "
                       "annualized return/vol, historical VaR 95/99 (positive "
                       "loss fractions), max drawdown (negative).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_anomalies",
        "description": "High-conviction anomalous market days (flagged by BOTH "
                       "the IsolationForest and the autoencoder), with scores.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer",
                                     "description": "Max days to return (default 20)"}},
        },
    },
    {
        "name": "get_graph_findings",
        "description": "Correlation network: nodes with eigenvector centrality "
                       "(systemic importance), community/cluster id, and a "
                       "correlation-profile shift z-score (recent 63d vs long-run).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_stress_scenario",
        "description": "Recompute portfolio metrics under a named stress "
                       "scenario applied to the full history.",
        "input_schema": {
            "type": "object",
            "properties": {"scenario": {
                "type": "string",
                "enum": list(stress.SCENARIOS),
                "description": "Which scenario to run",
            }},
            "required": ["scenario"],
        },
    },
    {
        "name": "compare_all_scenarios",
        "description": "Baseline vs every stress scenario in one table.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_fundamentals",
        "description": "Financial-statement ratios (liquidity, solvency, "
                       "profitability, efficiency) and DuPont ROE decomposition "
                       "per name, from SEC EDGAR 10-K filings.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forensic_scores",
        "description": "Forensic-accounting screens per name: Altman Z "
                       "(bankruptcy risk), Piotroski F (0-9 quality), Beneish M "
                       "(earnings-manipulation), accruals, plus a Benford's-Law "
                       "digit-conformity check across all reported figures.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_factor_model",
        "description": "Fama-French 5 + momentum regression: portfolio factor "
                       "loadings (betas) with t-stats, R², and factor-adjusted "
                       "annualized alpha, plus per-name loadings.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_optimal_portfolios",
        "description": "Markowitz optimisation: return/vol/Sharpe and weights for "
                       "min-variance, max-Sharpe, risk-parity and the equal-weight "
                       "baseline.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_distance_to_default",
        "description": "Merton (1974) structural credit model per name: "
                       "distance-to-default (asset-vol standard deviations above the "
                       "default point), implied 1-year default probability, and "
                       "leverage. The market's read on distress — the companion to "
                       "the accounting-based Altman Z. Banks are excluded (no "
                       "default-point tag).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_analog_days",
        "description": "The k historical days most similar to TODAY (KNN over "
                       "vol/drawdown/momentum/dispersion features) with the "
                       "realised forward-month return that followed each — an "
                       "empirical answer to 'given days like today, what tended "
                       "to happen next?'",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_tactical_backtest",
        "description": "Walk-forward backtest where the stress classifier gates "
                       "the allocation (min-variance when P(stress) is high, the "
                       "aggressive book otherwise): out-of-sample return/vol/"
                       "Sharpe/max-drawdown/Calmar for the gated and ungated "
                       "books. Shows where the ML overlay adds value (the "
                       "concentrated max-Sharpe book) and where it does not "
                       "(the diversified 1/N book).",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------- LLM plumbing

def _client():
    """Anthropic client if a key is configured, else None (template mode).

    We gate on the env var explicitly because this app also runs on shared
    hosts (Streamlit Cloud) where we only ever configure keys via env/secrets.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


def _model() -> str:
    return os.getenv("SENTINEL_LLM_MODEL", DEFAULT_MODEL)


def _text_of(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text")


# ---------------------------------------------------------------- memo

MEMO_PROMPT = """Write a Risk Memo for the portfolio using ONLY the data below.
Format as markdown with exactly these sections:

# Sentinel Risk Memo — {as_of}
## 1. Situation
## 2. Market Risk & Factor Exposure
## 3. Anomalies
## 4. Fundamental & Forensic Screens
## 5. Stress Tests & Allocation
## 6. Recommendation

Cover both lenses: market/factor risk (VaR, factor loadings, factor-adjusted
alpha) AND accounting health (ratios, Altman/Piotroski/Beneish/Benford screens,
plus the Merton distance-to-default as the market's structural credit read).
In section 5, note whether an optimised portfolio (max-Sharpe / min-variance)
would improve on the equal-weight book. If a data block is null, say the screen
was unavailable rather than inventing numbers. Keep it under 800 words,
decision-oriented, numbers quoted precisely.

DATA:
{data}
"""


def write_memo(context: dict | None = None, client=None) -> str:
    """The Risk Memo, as markdown. LLM-written when configured; otherwise a
    deterministic template so the pipeline never breaks."""
    context = context or gather_context()
    client = client or _client()
    if client is None:
        return _template_memo(context)

    response = client.messages.create(
        model=_model(),
        max_tokens=MEMO_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": MEMO_PROMPT.format(
            as_of=context["as_of"], data=json.dumps(context, default=str))}],
    )
    return _text_of(response)


def _template_memo(ctx: dict) -> str:
    """No-LLM fallback: same structure, filled from the computed numbers.
    Each depth lens degrades to a one-line 'unavailable' note if absent."""
    port = ctx["risk_summary"]["PORTFOLIO"]
    stress_rows = "\n".join(
        f"| {name} | {row['ann_vol']:.1%} | {row['var_95']:.2%} | {row['max_drawdown']:.1%} |"
        for name, row in ctx["stress"].items()
    )
    anomaly_dates = ", ".join(ctx["anomalies"]["agreed_dates"][-6:]) or "none"
    worst = max(ctx["stress"], key=lambda k: ctx["stress"][k]["var_95"])

    # today's regime-aware VaR vs the full-sample one — .get() so the minimal
    # hermetic test context (no ewma column) still renders.
    ewma_v = port.get("ewma_var_95")
    ewma_clause = (
        f" Today's regime-aware EWMA VaR95 is {ewma_v:.2%}, versus the full-sample "
        f"{port['var_95']:.2%} — the live read on whether risk is currently "
        f"elevated." if ewma_v is not None else "")

    factor = ctx.get("factor_model")
    factor_line = (
        f"A Fama-French 5+momentum regression explains R²={factor['r2']:.0%} of "
        f"return variance: market beta {factor['betas']['mkt_rf']:.2f}, and a "
        f"factor-adjusted alpha of {factor['alpha_ann']:+.1%} "
        f"(t={factor['alpha_t']:.1f}) survives stripping all six factors."
        if factor else "Factor decomposition unavailable.")

    fund, forensic = ctx.get("fundamentals"), ctx.get("forensic")
    if fund and forensic:
        distress = ", ".join(forensic["altman_distress"]) or "none"
        flags = ", ".join(forensic["beneish_flags"]) or "none"
        strong = ", ".join(forensic["piotroski_strong"]) or "none"
        fundamental_block = (
            f"Median ROE {fund['median_roe']:.0%}, net margin "
            f"{fund['median_net_margin']:.0%}, debt/equity "
            f"{fund['median_debt_to_equity']:.2f}. Altman distress: {distress}. "
            f"High Piotroski quality (F>=7): {strong}. Beneish manipulation flags: "
            f"{flags} (rapid-growth names can false-positive). Benford first-digit "
            f"conformity across all filings: {forensic['benford']['verdict']}.")
    else:
        fundamental_block = "Fundamental and forensic screens unavailable (EDGAR)."

    credit = ctx.get("credit")
    # ASCII only (no sigma glyph): this text is printed to the Windows console
    # and rendered into a reportlab PDF, neither of which handles non-cp1252.
    credit_line = (
        f"Structural distance-to-default (Merton): the thinnest cushion is "
        f"{credit['thinnest_name']}, sitting {credit['thinnest_dd']:.1f} standard "
        f"deviations above its default point (median {credit['median_dd']:.1f}) — "
        f"the market's read on the same distress Altman scores from the books."
        if credit else "Structural distance-to-default unavailable.")

    alloc = ctx.get("allocation")
    if alloc:
        s = alloc["stats"]
        alloc_line = (
            f"The equal-weight book runs a Sharpe of {s['Equal-weight']['sharpe']:.2f}. "
            f"A max-Sharpe tilt would raise that to {s['Max-Sharpe']['sharpe']:.2f}; "
            f"a min-variance mix cuts volatility to {s['Min-variance']['vol']:.1%} "
            f"(vs {s['Equal-weight']['vol']:.1%} equal-weight).")
    else:
        alloc_line = "Portfolio optimisation unavailable."

    return f"""# Sentinel Risk Memo — {ctx['as_of']}

*(Template memo — set ANTHROPIC_API_KEY for the full AI-written analysis.)*

## 1. Situation
Equal-weight portfolio of {len(ctx['tickers'])} names ({', '.join(ctx['tickers'])}).
Annualized vol {port['ann_vol']:.1%}, VaR95 {port['var_95']:.2%}/day, worst
historical drawdown {port['max_drawdown']:.1%}.

## 2. Market Risk & Factor Exposure
Portfolio vol is below every single constituent — diversification is working.
Systemic risk concentrates in the highest-centrality names in the correlation
network; the tech cluster moves as one block.{ewma_clause} {factor_line}

## 3. Anomalies
Both detectors (IsolationForest + autoencoder) currently agree on
{len(ctx['anomalies']['agreed_dates'])} anomalous days. Most recent agreed
anomalies: {anomaly_dates}.

## 4. Fundamental & Forensic Screens
{fundamental_block}

{credit_line}

## 5. Stress Tests & Allocation
| scenario | ann_vol | VaR95 | max drawdown |
|---|---|---|---|
{stress_rows}

{alloc_line}

## 6. Recommendation
The binding constraint is the systemic scenario ({worst}): diversification
across sectors does not survive correlated drawdowns. Maintain the equal-weight
structure, monitor the anomaly agreement signal, watch any name flagged by the
forensic screens, and size exposure to survive the worst-case drawdown above.
"""


# ---------------------------------------------------------------- ask (agentic)

def ask(question: str, client=None, max_turns: int = MAX_TOOL_TURNS) -> str:
    """Answer a free-text question by letting the model call our tools."""
    client = client or _client()
    if client is None:
        return ("LLM not configured (set ANTHROPIC_API_KEY in .env). "
                "The dashboard's Overview/Stress tabs show all computed metrics.")

    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=_model(),
            max_tokens=ASK_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            return _text_of(response)

        # append assistant turn (incl. tool_use + thinking blocks) verbatim,
        # then answer every tool call in ONE user message.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                content = TOOL_FUNCS[block.name](block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content})
            except Exception as exc:  # tool errors go back to the model, not up
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})

    return "I couldn't complete the analysis within the tool-call limit."


if __name__ == "__main__":
    from src.agent.report import save_markdown, save_pdf

    memo = write_memo()
    print(memo)
    print("\nSaved:", save_markdown(memo), "and", save_pdf(memo))
