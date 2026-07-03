"""Agent end-to-end with mocked LLM responses — no network, no API key."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent import memo
from src.agent.report import save_markdown, save_pdf


# ---------------------------------------------------------------- fakes

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_id="tu_1", tool_input=None):
    return SimpleNamespace(type="tool_use", name=name, id=tool_id,
                           input=tool_input or {})


class FakeClient:
    """Stands in for anthropic.Anthropic — returns scripted responses."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self._responses)


@pytest.fixture
def fake_returns(monkeypatch):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 3)), columns=["A", "B", "C"],
                      index=pd.bdate_range("2024-01-01", periods=300))
    monkeypatch.setattr(memo, "_returns", lambda: df)
    return df


@pytest.fixture
def context():
    """Minimal hand-built context — keeps template tests hermetic."""
    row = {"ann_return": 0.1, "ann_vol": 0.15, "var_95": 0.015,
           "var_99": 0.03, "max_drawdown": -0.25}
    return {
        "as_of": "2026-07-02",
        "tickers": ["A", "B"],
        "risk_summary": {"A": row, "B": row, "PORTFOLIO": row},
        "anomalies": {"if_flags": 4, "ae_flags": 4, "agreed_dates": ["2020-03-16"]},
        "graph": {"nodes": [], "edges": []},
        "stress": {"baseline": {"ann_return": 0.1, "ann_vol": 0.15,
                                "var_95": 0.015, "max_drawdown": -0.25},
                   "market_crash_2008_style": {"ann_return": -0.3, "ann_vol": 0.4,
                                               "var_95": 0.05, "max_drawdown": -0.9}},
        "scenario_descriptions": {},
    }


# ---------------------------------------------------------------- memo

def test_template_memo_without_llm(context):
    text = memo.write_memo(context=context, client=None)
    assert "# Sentinel Risk Memo — 2026-07-02" in text
    assert "## 5. Recommendation" in text
    assert "15.0%" in text          # ann_vol made it into prose
    assert "market_crash_2008_style" in text


def test_llm_memo_uses_client(context, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = FakeClient([SimpleNamespace(
        stop_reason="end_turn", content=[text_block("# Memo\nAll good.")])])
    text = memo.write_memo(context=context, client=client)
    assert text == "# Memo\nAll good."
    assert client.requests[0]["system"] == memo.SYSTEM


# ---------------------------------------------------------------- ask loop

def test_ask_without_llm_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in memo.ask("What is the VaR?")


def test_ask_executes_tool_then_answers(fake_returns):
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use",
                        content=[tool_block("get_risk_summary")]),
        SimpleNamespace(stop_reason="end_turn",
                        content=[text_block("Portfolio VaR95 is 1.5%.")]),
    ])
    answer = memo.ask("What is the portfolio VaR?", client=client)
    assert answer == "Portfolio VaR95 is 1.5%."
    # second request must carry the tool result back
    followup = client.requests[1]["messages"]
    results = followup[-1]["content"]
    assert results[0]["type"] == "tool_result"
    assert "PORTFOLIO" in results[0]["content"]


def test_ask_returns_tool_errors_to_model(fake_returns):
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use",
                        content=[tool_block("run_stress_scenario",
                                            tool_input={"scenario": "nope"})]),
        SimpleNamespace(stop_reason="end_turn",
                        content=[text_block("That scenario doesn't exist.")]),
    ])
    answer = memo.ask("Run the nope scenario", client=client)
    assert answer == "That scenario doesn't exist."
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_ask_stops_at_max_turns(fake_returns):
    looping = SimpleNamespace(stop_reason="tool_use",
                              content=[tool_block("get_risk_summary")])
    client = FakeClient([looping] * 3)
    answer = memo.ask("loop forever", client=client, max_turns=3)
    assert "tool-call limit" in answer


# ---------------------------------------------------------------- reports

def test_reports_render(tmp_path, context):
    text = memo.write_memo(context=context, client=None)
    md = save_markdown(text, tmp_path / "memo.md")
    pdf = save_pdf(text, tmp_path / "memo.pdf")
    assert md.read_text(encoding="utf-8").startswith("# Sentinel Risk Memo")
    assert pdf.stat().st_size > 1000  # non-trivial PDF with table + headings
