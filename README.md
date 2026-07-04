# Sentinel 🛰️

*A junior quant risk analyst, automated.*

![Python](https://img.shields.io/badge/python-3.11+-00E5A0?logo=python&logoColor=white&labelColor=0B0E14)
![Tests](https://img.shields.io/badge/tests-34%20passing-00E5A0?labelColor=0B0E14)
![FastAPI](https://img.shields.io/badge/API-FastAPI-00E5A0?logo=fastapi&logoColor=white&labelColor=0B0E14)
![PyTorch](https://img.shields.io/badge/ML-PyTorch%20%2B%20sklearn-00E5A0?logo=pytorch&logoColor=white&labelColor=0B0E14)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-00E5A0?logo=streamlit&logoColor=white&labelColor=0B0E14)

**[Showcase site →](https://sentinel-eight-xi.vercel.app)** · **[Live dashboard →](https://sentinel-risk.streamlit.app)** <!-- TODO: real Streamlit URL after deploy -->

**Sentinel is an open-source agentic financial risk management engine**: it
ingests market data, computes portfolio risk metrics (volatility, historical
Value-at-Risk, max drawdown), detects anomalous market days with machine
learning (a PyTorch autoencoder + IsolationForest), maps systemic risk as a
correlation network, runs macro stress-test scenarios, and uses an LLM agent
with tool access to the whole engine to write the analyst risk memo — served
via a FastAPI REST API, a Docker container, and a Streamlit dashboard.

---

## The problem

Every risk desk runs the same daily loop: pull prices, recompute metrics,
eyeball charts for anything weird, stress the book, and write a memo about it —
hours of skilled-analyst time producing a report about *yesterday*. The
interesting question isn't whether each step can be automated (it can); it's
whether the **whole loop** — including the judgment-flavored parts like
"which days deserved attention?" and "what would I tell the committee?" —
can run end-to-end without a human in the middle.

## The approach

Sentinel is that loop as software. It ingests market data, computes the
classic risk battery, then layers on three things a junior analyst would
actually be asked for:

1. **Anomaly detection with two independent ML models** — an IsolationForest
   baseline and a PyTorch autoencoder that learns to reconstruct "normal"
   market days. Neither is told anything about events; agreement between them
   is the high-conviction signal.
2. **Structure, not just numbers** — a correlation network (NetworkX) that
   maps who moves with whom, ranks systemic importance by eigenvector
   centrality, and flags names whose correlation profile drifts from its
   long-run pattern.
3. **A GenAI agent that closes the loop** — given tool access to the entire
   engine (not a prompt full of pasted numbers), it answers free-text
   questions and writes the structured risk memo, ending in a recommendation.

Every non-obvious modeling choice is documented in the code with a short
"why" comment, and every layer has tests with known-input sanity checks.

## Architecture

```
 yfinance ──▶ DuckDB warehouse ──▶ risk metrics ─┐
              (prices, returns     anomaly (IF + AE)├──▶ Claude agent ──▶ risk memo (MD/PDF)
               derived in SQL)     graph (NetworkX) │    (tool use)
                                   stress engine  ─┘         │
                                        │                    │
                                        ▼                    ▼
                                   FastAPI (Docker) ◀──── /ask, /memo
                                        │
                                        ▼
                                Streamlit dashboard · Next.js showcase site
```

## Key findings

**1. Both anomaly models independently rediscovered the two real stress events
in the sample — from raw returns alone.** 43 flags each (calibrated to a 2%
base rate), 14 days of agreement: the March–April 2020 COVID crash cluster and
April 9, 2025 (the tariff-pause day, one of the largest single-day moves since
2008). No event data, no labels, no news feed.

![Anomaly detection timeline](docs/anomalies.png)

**2. The correlation network recovers the economy's sector structure with zero
labels** — greedy-modularity communities on daily-return correlations split the
book cleanly into tech, cyclicals, and defensives. Centrality shows systemic
risk concentrating in the tech/cyclical core, while WMT and NEE sit at the
periphery: that's where the diversification actually lives.

![Correlation network](docs/network.png)

**3. Diversification works — until it doesn't.** The equal-weight portfolio's
volatility (18.7% ann.) is below *every single constituent*, and daily VaR95 is
1.7% vs 3–5% for single names. But the stress engine's 2008-style scenario
(vol ×2.5, −0.2%/day systemic drift) produces a **−97% drawdown**: when
correlations go to one, sector diversification is no defense. The agent's memo
correctly identifies this as the binding constraint.

| scenario | ann. vol | VaR 95 | max drawdown |
|---|---|---|---|
| baseline | 18.7% | 1.69% | −30.0% |
| rate_shock | 24.3% | 2.26% | −38.8% |
| sector_shock_tech | 26.2% | 2.47% | −49.2% |
| **market_crash_2008_style** | **46.7%** | **4.56%** | **−97.2%** |

## What each component does

| Component | Role |
|---|---|
| [`src/ingest/market.py`](src/ingest/market.py) | Daily OHLCV via yfinance; parquet cache; gap handling that bridges halts but never invents pre-listing history |
| [`src/warehouse/duck.py`](src/warehouse/duck.py) | DuckDB warehouse; returns derived **in SQL** so every consumer shares one source of truth |
| [`src/models/risk.py`](src/models/risk.py) | Annualized return/vol, rolling vol, historical VaR 95/99, max drawdown, correlations — each with the finance explained |
| [`src/models/anomaly.py`](src/models/anomaly.py) | IsolationForest + autoencoder (20→8→3→8→20) over per-name returns and rolling vol; agreement flag |
| [`src/models/graph.py`](src/models/graph.py) | Correlation network, per-component eigenvector centrality, communities, 63-day correlation-shift detector |
| [`src/models/stress.py`](src/models/stress.py) | Parameterized scenario engine (vol multiplier + drift shocks) replayed over history |
| [`src/agent/memo.py`](src/agent/memo.py) | Claude agent with 5 tools over the engine; writes the memo, answers questions; deterministic fallback without a key |
| [`src/api/main.py`](src/api/main.py) | FastAPI: `/metrics`, `/stress`, `/anomalies`, `/ask`, `/memo` — typed responses, self-bootstrapping data |
| [`dashboard/`](dashboard/app.py) | Streamlit risk terminal — dark, mint-accent, five tabs including "Ask the Agent" |
| [`site/`](site/) | Next.js + Tailwind + framer-motion showcase page (Vercel) |
| [`notebooks/`](notebooks/) | The narrative analysis: executed notebooks with key-findings cells |

## How to run

```sh
# 1. Environment
python -m venv .venv && .venv\Scripts\activate     # Windows (source .venv/bin/activate on unix)
pip install -r requirements.txt
copy .env.example .env                              # optional: add ANTHROPIC_API_KEY for the AI memo

# 2. Pick your surface
pytest                                              # 34 tests, no network needed
uvicorn src.api.main:app --port 8000                # API  -> http://localhost:8000/docs
streamlit run dashboard/app.py                      # dashboard -> http://localhost:8501
docker compose up --build                           # containerized API

# 3. Showcase site
cd site && npm install && npm run dev               # -> http://localhost:3000
```

Everything bootstraps its own data on first run (yfinance → DuckDB). Without
an `ANTHROPIC_API_KEY` the agent degrades gracefully to a templated memo
filled with the real computed numbers.

## Limitations & how I'd scale it

- **Batch, not streaming.** Data refreshes on demand. The natural next step is
  a Kafka ingestion topic with Spark (or Flink) computing rolling metrics
  continuously, DuckDB swapped for a real warehouse, and anomaly scoring moved
  to an online model served behind the same API contract.
- **Single asset class, daily bars.** The metric layer is shape-agnostic
  (anything with a returns matrix works), so rates/FX/crypto and intraday bars
  are config away — the interesting work is recalibrating the anomaly base rate.
- **Macro context is thin.** FRED and SEC EDGAR loaders are the planned next
  ingestion targets, so the agent can condition its memo on rates and filings,
  not just prices.
- **Stress scenarios are stylized.** Vol-multiplier + drift shocks are
  transparent and committee-explainable, but historical bootstrapping or
  factor-model shocks would be the production upgrade.

## FAQ

**What is Sentinel?**
An agentic quantitative risk engine — a Python system that automates the
daily workflow of a junior quant risk analyst: risk metrics, ML anomaly
detection, correlation network analysis, stress testing, and a written,
decision-oriented risk memo generated by an LLM agent.

**How does the AI agent work?**
It's genuine tool use, not a prompt full of pasted numbers: a Claude model is
given five callable tools over the engine (risk summary, anomalies, network
findings, stress scenarios) and decides which to call to answer a question or
write the memo. Without an API key it degrades to a deterministic template
memo, so the whole system runs offline.

**How does the ML anomaly detection work?**
Two independent models score every trading day on each name's return and
rolling volatility (z-scored): an IsolationForest baseline, and a PyTorch
autoencoder that learns to reconstruct "normal" market days and flags high
reconstruction error. Days where both agree are the high-conviction signal —
in this sample, exactly the COVID crash and the 2025 tariff shock.

**How is Value-at-Risk calculated?**
Historical (non-parametric) VaR: the empirical quantile of realized daily
returns, reported as a positive loss fraction — no normality assumption, so
fat tails are captured as observed.

**Can I use it with my own portfolio?**
Yes — set `SENTINEL_TICKERS` in `.env` to any list of Yahoo Finance symbols.
Every layer (metrics, anomaly models, network, stress, agent, API, dashboard)
adapts to whatever returns matrix comes out of the warehouse.

## Tech stack

Python 3.11+ · pandas · yfinance · DuckDB · scikit-learn · PyTorch · NetworkX ·
Anthropic API (claude-opus-4-8) · FastAPI · Docker · Streamlit · Plotly ·
Next.js · Tailwind · framer-motion

---

Built by **Shreshtha Rawat** — [GitHub](https://github.com/nightraven4545) · [LinkedIn](https://www.linkedin.com/in/shreshtha-rawat) <!-- TODO: real LinkedIn -->
