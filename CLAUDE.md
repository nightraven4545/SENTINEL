# Sentinel — Agentic Quant Risk & Anomaly Engine

*"A junior quant risk analyst, automated."*

## What Sentinel is
An agentic financial risk & anomaly engine: it ingests market data, computes risk
metrics, detects anomalies with ML, runs stress-test scenarios, and has a GenAI
agent that interprets results and writes an analyst risk memo. Served via FastAPI,
containerized with Docker, presented in a Streamlit dashboard, and showcased on a
Next.js/Vercel site.

## Goal
Job-portfolio project reverse-engineered from a McKinsey quant/AI role. It must read
as senior work: clean architecture, real analysis, a decision-oriented story, and a
polished UI. Prioritize clarity, correctness, and presentation over cleverness. The
author is rusty — explain non-obvious choices briefly in comments and keep code
readable.

## Tech stack (do not add anything heavier without asking)
- Python 3.11, pandas, numpy
- Data: yfinance (market), FRED (macro), SEC EDGAR (filings) — all free, no paid keys
- Storage: DuckDB (local analytical DB)
- ML: scikit-learn (IsolationForest), PyTorch (autoencoder), NetworkX (graph)
- Serving: FastAPI + Uvicorn, Dockerfile
- Dashboard: Streamlit (Streamlit Community Cloud)
- Showcase site: Next.js + Tailwind + framer-motion (Vercel)
- Config via .env; never hardcode secrets. Include .env.example.

## Repo structure
```
README.md            (case-study style — written LAST, after we have results)
CLAUDE.md
requirements.txt
.env.example
.gitignore
data/                (gitignored except .gitkeep + tiny samples)
src/
  ingest/            (yfinance, fred, edgar loaders)
  warehouse/         (duckdb schema + load)
  models/            (risk metrics, anomaly, graph, stress)
  agent/             (LLM memo writer)
  api/               (FastAPI app)
notebooks/           (analysis notebooks with narrative)
dashboard/           (streamlit app)
site/                (next.js vercel showcase — separate app)
reports/             (generated PDF/memo output)
tests/               (pytest)
```

## Coding standards
- Type hints, docstrings, small functions, clear names.
- Separate config/inputs from logic. No magic numbers — name constants.
- Every module runnable + a matching test where it makes sense.
- Write a short "why" comment on any non-trivial financial or ML choice.

## Design language (both UIs must match — "Bloomberg terminal, but modern")
- Background `#0B0E14` / cards `#12161F`
- Text `#E6EDF3`, muted `#8B949E`
- Single accent: mint `#00E5A0`
- Fonts: Inter (text) + a monospace (numbers/code)
- Charts: Plotly `plotly_dark`, mint for the key series, thin gridlines, no clutter
- Motion: subtle fade-and-rise on scroll, gentle hover lift — never flashy

## Build plan (one session at a time, commit after each)
1. Data + risk metrics + anomaly detection
2. Graph + stress-testing
3. GenAI agent (memo writer + tool-use)
4. FastAPI + Docker
5. Themed Streamlit dashboard
6. Vercel showcase site
7. Case-study README + polish
