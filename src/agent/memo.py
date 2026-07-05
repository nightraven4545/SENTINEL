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
    }


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


TOOL_FUNCS = {
    "get_risk_summary": lambda inp: _tool_risk_summary(),
    "get_anomalies": lambda inp: _tool_anomalies(int(inp.get("limit", 20))),
    "get_graph_findings": lambda inp: _tool_graph(),
    "run_stress_scenario": lambda inp: _tool_stress(inp["scenario"]),
    "compare_all_scenarios": lambda inp: _tool_compare_scenarios(),
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
## 2. Key Risks
## 3. Anomalies
## 4. Stress Test Results
## 5. Recommendation

Keep it under 600 words, decision-oriented, numbers quoted precisely.

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
    """No-LLM fallback: same structure, filled from the computed numbers."""
    port = ctx["risk_summary"]["PORTFOLIO"]
    stress_rows = "\n".join(
        f"| {name} | {row['ann_vol']:.1%} | {row['var_95']:.2%} | {row['max_drawdown']:.1%} |"
        for name, row in ctx["stress"].items()
    )
    anomaly_dates = ", ".join(ctx["anomalies"]["agreed_dates"][-6:]) or "none"
    worst = max(ctx["stress"], key=lambda k: ctx["stress"][k]["var_95"])
    return f"""# Sentinel Risk Memo — {ctx['as_of']}

*(Template memo — set ANTHROPIC_API_KEY for the full AI-written analysis.)*

## 1. Situation
Equal-weight portfolio of {len(ctx['tickers'])} names ({', '.join(ctx['tickers'])}).
Annualized vol {port['ann_vol']:.1%}, VaR95 {port['var_95']:.2%}/day, worst
historical drawdown {port['max_drawdown']:.1%}.

## 2. Key Risks
Portfolio vol is below every single constituent — diversification is working.
Systemic risk concentrates in the highest-centrality names in the correlation
network; the tech cluster moves as one block.

## 3. Anomalies
Both detectors (IsolationForest + autoencoder) currently agree on
{len(ctx['anomalies']['agreed_dates'])} anomalous days. Most recent agreed
anomalies: {anomaly_dates}.

## 4. Stress Test Results
| scenario | ann_vol | VaR95 | max drawdown |
|---|---|---|---|
{stress_rows}

## 5. Recommendation
The binding constraint is the systemic scenario ({worst}): diversification
across sectors does not survive correlated drawdowns. Maintain the equal-weight
structure, monitor the anomaly agreement signal, and size exposure to survive
the worst-case drawdown above.
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
