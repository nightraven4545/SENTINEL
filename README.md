# Sentinel

*Agentic quant risk & anomaly engine — "a junior quant risk analyst, automated."*

> 🚧 Build in progress. The case-study README is written last (Session 7), once
> there are hero charts and live demo links. See [CLAUDE.md](CLAUDE.md) for the
> full plan.

## API

Run locally:

```sh
pip install -r requirements.txt
uvicorn src.api.main:app --port 8000
```

Or with Docker (bootstraps its own market data on first start):

```sh
docker compose up --build
```

Interactive docs at **http://localhost:8000/docs**. Endpoints:

```sh
# Portfolio + per-ticker risk metrics
curl http://localhost:8000/metrics

# Run a named stress scenario
curl -X POST http://localhost:8000/stress \
  -H "content-type: application/json" \
  -d '{"scenario": "market_crash_2008_style"}'

# Anomalous market days (both ML detectors)
curl "http://localhost:8000/anomalies?flagged_only=true"

# Ask the agent a question (tool-use over the risk engine)
curl -X POST http://localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "Which scenario hurts the portfolio most, and why?"}'

# Generate the risk memo (LLM if ANTHROPIC_API_KEY is set, template otherwise)
curl http://localhost:8000/memo
```

Configuration via `.env` — see [.env.example](.env.example). The agent
endpoints work without an API key (deterministic fallback), and everything
else is fully local: yfinance → DuckDB → scikit-learn/PyTorch/NetworkX.
