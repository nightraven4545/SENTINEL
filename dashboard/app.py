"""Sentinel dashboard — Streamlit front-end over the risk engine.

Calls the src/ modules directly (no API dependency) so it runs anywhere the
repo runs, including Streamlit Community Cloud. Heavy computations are cached
per session; the warehouse bootstraps itself on first run.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- design tokens
MINT = "#00E5A0"
BG = "#0E1117"
CARD = "#161B22"
BORDER = "#21262D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
AMBER = "#E3B341"
RED = "#F85149"
CLUSTER_COLORS = [MINT, "#58A6FF", AMBER, "#F778BA"]

st.set_page_config(page_title="Sentinel — Risk Terminal", page_icon="🛰️",
                   layout="wide")

st.markdown(f"""<style>
.block-container {{ padding-top: 2.2rem; }}
.sentinel-header {{ display:flex; align-items:baseline; gap:14px;
    border-bottom:1px solid {BORDER}; padding-bottom:14px; margin-bottom:6px; }}
.sentinel-header .name {{ font-size:1.7rem; font-weight:700; letter-spacing:.14em;
    color:{TEXT}; }}
.sentinel-header .dot {{ color:{MINT}; }}
.sentinel-header .tag {{ color:{MUTED}; font-size:.85rem; }}
.kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
    padding:16px 20px 14px; }}
.kpi .label {{ color:{MUTED}; font-size:.7rem; letter-spacing:.1em;
    text-transform:uppercase; }}
.kpi .value {{ color:{TEXT}; font-size:1.85rem; font-weight:650; line-height:1.25; }}
.kpi .delta {{ font-size:.75rem; }}
.kpi .up {{ color:{MINT}; }} .kpi .down {{ color:{RED}; }} .kpi .flat {{ color:{MUTED}; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached data

@st.cache_data(show_spinner="Loading warehouse (first run fetches market data)…")
def load_returns() -> pd.DataFrame:
    from src.warehouse.duck import ensure_loaded, returns_wide
    ensure_loaded()  # fresh deploy (e.g. Streamlit Cloud) bootstraps itself
    return returns_wide()


@st.cache_data(show_spinner="Training anomaly models…")
def load_anomalies() -> pd.DataFrame:
    from src.models.anomaly import detect_anomalies
    return detect_anomalies(load_returns())


@st.cache_data(show_spinner="Building correlation network…")
def load_graph() -> dict:
    from src.models.graph import graph_payload
    return graph_payload(load_returns())


@st.cache_data(show_spinner="Running stress scenarios…")
def load_stress() -> pd.DataFrame:
    from src.models.stress import compare
    return compare(load_returns())


# ---------------------------------------------------------------- helpers

def themed(fig: go.Figure, height: int = 420) -> go.Figure:
    """One consistent chart language: dark, thin gridlines, no clutter."""
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=BORDER, gridwidth=0.5, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, gridwidth=0.5, zeroline=False)
    return fig


def kpi(col, label: str, value: str, delta: str = "", direction: str = "flat"):
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="delta {direction}">{delta}</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------- header + data

st.markdown(
    '<div class="sentinel-header"><span class="name">SENTINEL<span class="dot">▮</span></span>'
    '<span class="tag">a junior quant risk analyst, automated</span></div>',
    unsafe_allow_html=True)

returns = load_returns()
from src.models import risk  # noqa: E402  (after sys.path fix)

port = risk.portfolio_returns(returns)
tabs = st.tabs(["Overview", "Deep Dive", "Anomalies", "Network", "Stress Test",
                "Fundamentals", "Forensic", "Factors", "Allocation", "ML Models",
                "Ask the Agent"])


@st.cache_data(show_spinner="Loading benchmark…")
def load_benchmark() -> pd.Series | None:
    try:
        from src.ingest.market import fetch_benchmark
        return fetch_benchmark()
    except Exception:
        return None  # CAPM cards degrade gracefully if SPY is unavailable


@st.cache_data(show_spinner="Loading SEC filings…")
def load_fundamentals():
    """Latest-FY ratios + DuPont from EDGAR. Returns None if both SEC and the
    bundled snapshot are unavailable, so the tab can degrade gracefully."""
    try:
        from src.ingest.edgar import latest_fundamentals
        from src.models.fundamentals import dupont, ratios
        w = latest_fundamentals()
        return ratios(w), dupont(w)
    except Exception:
        return None


@st.cache_data(show_spinner="Running forensic screens…")
def load_forensic():
    """Forensic scores + Benford. Altman's X4 needs market value of equity, so
    market cap = latest close × shares outstanding (both from the cloud-safe
    loaders). Returns None if the data can't be assembled."""
    try:
        from src.ingest.edgar import fetch_fundamentals, latest_fundamentals
        from src.ingest.market import fetch_prices
        from src.models import forensic as fx
        long = fetch_fundamentals()
        last_close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
        shares = latest_fundamentals(long).get("shares")
        mcap = (shares * last_close).dropna() if shares is not None else None
        return {
            "summary": fx.forensic_summary(long, mcap),
            "altman": fx.altman_z(long, mcap),
            "beneish": fx.beneish_m(long),
            "benford": fx.benford_distribution(long),
            "benford_mad": fx.benford_mad(long),
        }
    except Exception:
        return None


@st.cache_data(show_spinner="Solving Merton model…")
def load_credit():
    """Merton distance-to-default per name — the market's structural view of the
    same distress Altman reads off the balance sheet. None if inputs unavailable."""
    try:
        from src.ingest.edgar import latest_fundamentals
        from src.ingest.market import fetch_prices
        from src.models.credit import default_point, distance_to_default
        w = latest_fundamentals()
        last_close = fetch_prices().sort_values("date").groupby("ticker")["close"].last()
        shares = w.get("shares")
        if shares is None:
            return None
        mcap = (shares * last_close).dropna()
        return distance_to_default(returns, mcap, default_point(w))
    except Exception:
        return None


@st.cache_data(show_spinner="Fetching Fama-French factors…")
def load_factor_model():
    """Portfolio factor regression + per-name loadings, or None if the factor
    library (and its committed snapshot) can't be reached."""
    try:
        from src.ingest.factors import fetch_factors
        from src.models.factors import factor_loadings, portfolio_factor_model
        f = fetch_factors()
        return portfolio_factor_model(returns, f), factor_loadings(returns, f)
    except Exception:
        return None


@st.cache_data(show_spinner="Optimising portfolios…")
def load_optimization():
    """Efficient frontier, optimal portfolios, and per-asset risk/return."""
    try:
        import numpy as _np
        from src.models.optimize import (annualized_moments, compare_portfolios,
                                          efficient_frontier)
        from src.models.risk import RISK_FREE_RATE
        stats, weights = compare_portfolios(returns)
        mu, cov = annualized_moments(returns)
        asset_vol = pd.Series(_np.sqrt(_np.diag(cov.to_numpy())), index=returns.columns)
        return {"stats": stats, "weights": weights,
                "ef": efficient_frontier(returns, 50),
                "mu": mu, "asset_vol": asset_vol, "rf": RISK_FREE_RATE}
    except Exception:
        return None


@st.cache_data(show_spinner="Training stress classifier…")
def load_classifier():
    """Supervised stress-day classifier (logistic + random forest). None on failure."""
    try:
        from src.models.classify import train_stress_classifier
        return train_stress_classifier(returns)
    except Exception:
        return None


@st.cache_data(show_spinner="Running PCA + KMeans…")
def load_clusters():
    """PCA statistical factors + KMeans peer clusters. None on failure."""
    try:
        from src.models.unsupervised import cluster_universe, pca_factors
        return {"pca": pca_factors(returns), "km": cluster_universe(returns)}
    except Exception:
        return None


# ---------------------------------------------------------------- Overview

with tabs[0]:
    anoms = load_anomalies()
    agreed = anoms[anoms["both_flag"]]

    full_vol = float(risk.annualized_vol(port))
    now_vol = float(risk.rolling_vol(port).iloc[-1])
    vol_dir = "down" if now_vol > full_vol else "up"  # hotter regime = red
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Portfolio vol (ann.)", f"{full_vol:.1%}",
        f"21d now: {now_vol:.1%}", vol_dir)
    kpi(c2, "VaR 95 (daily)", f"{float(risk.hist_var(port, 0.95)):.2%}",
        f"VaR 99: {float(risk.hist_var(port, 0.99)):.2%}", "flat")
    kpi(c3, "Max drawdown", f"{float(risk.max_drawdown(port)):.1%}",
        "worst peak-to-trough", "flat")
    last_anom = agreed.index.max().date() if len(agreed) else "—"
    kpi(c4, "Anomalies (agreed)", f"{len(agreed)}",
        f"last: {last_anom}", "flat")

    st.write("")
    wealth = (1 + returns).cumprod()
    port_wealth = (1 + port).cumprod()
    fig = go.Figure()
    for col in wealth.columns:
        fig.add_scatter(x=wealth.index, y=wealth[col], name=col,
                        line=dict(width=1, color=MUTED), opacity=0.4)
    fig.add_scatter(x=port_wealth.index, y=port_wealth, name="PORTFOLIO",
                    line=dict(width=2.5, color=MINT))
    fig.update_layout(title="Growth of $1 — constituents vs equal-weight portfolio")
    st.plotly_chart(themed(fig), width="stretch")

    roll = risk.rolling_vol(port)
    ewma = risk.ewma_vol(port)
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name="21d rolling vol",
                    line=dict(width=1.2, color=MUTED))
    fig.add_scatter(x=ewma.index, y=ewma, name=f"EWMA (λ={risk.EWMA_LAMBDA})",
                    line=dict(width=1.8, color=MINT))
    fig.add_hline(y=full_vol, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text="full-sample")
    fig.update_layout(title="Portfolio volatility (annualized) — 21d rolling vs EWMA conditional",
                      yaxis_tickformat=".0%")
    st.plotly_chart(themed(fig, 320), width="stretch")
    st.caption("EWMA decays the weight on past shocks geometrically (RiskMetrics "
               f"λ={risk.EWMA_LAMBDA}), so it tracks the regime smoothly where the "
               "equal-weighted rolling window steps and lags.")


# ---------------------------------------------------------------- Deep Dive

with tabs[1]:
    bench = load_benchmark()

    # -- risk-adjusted ratios + CAPM vs SPY
    c = st.columns(5)
    kpi(c[0], "Sharpe", f"{float(risk.sharpe_ratio(port)):.2f}",
        f"rf = {risk.RISK_FREE_RATE:.0%}", "flat")
    kpi(c[1], "Sortino", f"{float(risk.sortino_ratio(port)):.2f}",
        "downside-only vol", "flat")
    kpi(c[2], "Calmar", f"{float(risk.calmar_ratio(port)):.2f}",
        "return / |max DD|", "flat")
    if bench is not None:
        stats = risk.capm(port, bench)
        kpi(c[3], "Beta vs SPY", f"{stats['beta']:.2f}",
            f"R² {stats['r2']:.2f}", "flat")
        alpha_dir = "up" if stats["alpha_ann"] > 0 else "down"
        kpi(c[4], "Alpha (ann.)", f"{stats['alpha_ann']:+.1%}",
            f"IR {stats['information_ratio']:.2f} · TE {stats['tracking_error']:.1%}",
            alpha_dir)
    else:
        kpi(c[3], "Beta vs SPY", "—", "benchmark unavailable", "flat")
        kpi(c[4], "Alpha (ann.)", "—", "benchmark unavailable", "flat")

    st.write("")
    left, right = st.columns(2, gap="large")

    # -- VaR methodology comparison: three answers to "how bad is bad?"
    with left:
        methods = {
            "Historical": (risk.hist_var(port, 0.95), risk.hist_var(port, 0.99)),
            "Cornish-Fisher": (risk.cornish_fisher_var(port, 0.95),
                               risk.cornish_fisher_var(port, 0.99)),
            "Expected Shortfall": (risk.expected_shortfall(port, 0.95),
                                   risk.expected_shortfall(port, 0.99)),
            "EWMA (today)": (risk.ewma_var(port, 0.95), risk.ewma_var(port, 0.99)),
        }
        fig = go.Figure()
        fig.add_bar(x=list(methods), y=[float(v[0]) for v in methods.values()],
                    name="95%", marker_color=MINT)
        fig.add_bar(x=list(methods), y=[float(v[1]) for v in methods.values()],
                    name="99%", marker_color=MUTED)
        fig.update_layout(barmode="group", title="Daily tail risk — four methodologies",
                          yaxis_tickformat=".2%")
        st.plotly_chart(themed(fig, 380), width="stretch")
        st.caption("The first three read the whole sample; **EWMA (today)** uses "
                   "only the latest conditional volatility — a regime-aware VaR "
                   "that moves the moment markets turn.")
        bt = risk.kupiec_test(port)
        verdict = "✅ not rejected" if bt["passes"] else "❌ REJECTED"
        st.caption(f"**VaR95 backtest (Kupiec POF, rolling {risk.VAR_BACKTEST_WINDOW}d "
                   f"out-of-sample):** {bt['breaches']} breaches vs "
                   f"{bt['expected_breaches']} expected over {bt['observations']} days "
                   f"— model {verdict} (p = {bt['p_value']:.2f}).")

    # -- Euler risk decomposition: weight vs actual share of risk
    with right:
        contrib = risk.risk_contributions(returns).sort_values("pct_of_risk",
                                                               ascending=False)
        fig = go.Figure()
        fig.add_bar(x=contrib.index, y=contrib["weight"], name="capital weight",
                    marker_color=MUTED)
        fig.add_bar(x=contrib.index, y=contrib["pct_of_risk"], name="share of risk",
                    marker_color=MINT)
        fig.update_layout(barmode="group", yaxis_tickformat=".0%",
                          title="Capital vs risk — Euler decomposition of portfolio vol")
        st.plotly_chart(themed(fig, 380), width="stretch")
        top = contrib.iloc[0]
        st.caption(f"**{contrib.index[0]}** holds {top['weight']:.0%} of capital but "
                   f"contributes {top['pct_of_risk']:.0%} of portfolio risk — "
                   "equal weight is not equal risk.")

    # -- underwater plot: depth AND duration of drawdowns
    dd = risk.drawdown_series(port)
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd, fill="tozeroy", name="drawdown",
                    line=dict(width=1, color=MINT),
                    fillcolor="rgba(0,229,160,0.12)")
    fig.update_layout(title=f"Underwater plot — longest recovery: "
                            f"{risk.max_drawdown_duration(port)} trading days",
                      yaxis_tickformat=".0%")
    st.plotly_chart(themed(fig, 320), width="stretch")

    st.dataframe(
        risk.summary(returns).round(3).sort_values("sharpe", ascending=False),
        width="stretch", height=320)
    st.caption("Full risk battery per name: Sharpe/Sortino (rf = "
               f"{risk.RISK_FREE_RATE:.0%}), four tail measures (historical, "
               "Cornish-Fisher, expected shortfall, and today's EWMA), and higher "
               "moments. Negative skew + excess kurtosis = fatter left tails "
               "than the normal VaR assumes — exactly why Cornish-Fisher and "
               "ES sit above historical VaR.")


# ---------------------------------------------------------------- Anomalies

with tabs[2]:
    anoms = load_anomalies()
    aligned = port.loc[anoms.index]
    one_model = (anoms["if_flag"] | anoms["ae_flag"]) & ~anoms["both_flag"]

    fig = go.Figure()
    fig.add_scatter(x=aligned.index, y=aligned, mode="lines", name="daily return",
                    line=dict(width=0.8, color=MUTED))
    fig.add_scatter(x=aligned.index[one_model], y=aligned[one_model], mode="markers",
                    name="one model", marker=dict(color=AMBER, size=6))
    fig.add_scatter(x=aligned.index[anoms["both_flag"]], y=aligned[anoms["both_flag"]],
                    mode="markers", name="both models",
                    marker=dict(color=MINT, size=9,
                                line=dict(color=TEXT, width=0.5)))
    fig.update_layout(title="Portfolio returns — anomalous days highlighted",
                      yaxis_tickformat=".1%")
    st.plotly_chart(themed(fig), width="stretch")

    st.caption("IsolationForest + PyTorch autoencoder over per-name returns and "
               "rolling vol. Mint = both models agree (high conviction).")
    flagged = anoms[anoms["if_flag"] | anoms["ae_flag"]].sort_index(ascending=False)
    st.dataframe(flagged.round(4), width="stretch", height=320)


# ---------------------------------------------------------------- Network

with tabs[3]:
    payload = load_graph()
    nodes = pd.DataFrame(payload["nodes"]).set_index("id")

    fig = go.Figure()
    for e in payload["edges"]:
        a, b = nodes.loc[e["source"]], nodes.loc[e["target"]]
        fig.add_scatter(x=[a["x"], b["x"]], y=[a["y"], b["y"]],
                        mode="lines", line=dict(width=5 * e["weight"], color=MUTED),
                        opacity=0.3, hoverinfo="skip", showlegend=False)
    for cluster, grp in nodes.groupby("cluster"):
        fig.add_scatter(
            x=grp["x"], y=grp["y"], mode="markers+text", text=grp.index,
            textposition="top center", name=f"cluster {cluster}",
            textfont=dict(size=13, color=TEXT),
            customdata=grp[["centrality", "shift_z"]],
            hovertemplate="<b>%{text}</b><br>centrality %{customdata[0]:.2f}"
                          "<br>corr-shift z %{customdata[1]:.2f}<extra></extra>",
            marker=dict(size=22 + 34 * grp["centrality"],
                        color=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                        line=dict(width=grp["anomalous"].map({True: 3, False: 0}),
                                  color=MINT)))
    fig.update_layout(title=f"Correlation network — edges where corr > "
                            f"{payload['threshold']} · size = systemic centrality "
                            f"· color = community",
                      xaxis_visible=False, yaxis_visible=False)
    # pad ranges so labels/markers at layout extremes don't clip
    fig.update_xaxes(range=[nodes["x"].min() - 0.35, nodes["x"].max() + 0.35])
    fig.update_yaxes(range=[nodes["y"].min() - 0.3, nodes["y"].max() + 0.35])
    st.plotly_chart(themed(fig, 560), width="stretch")

    st.caption("Communities are discovered from returns alone (no sector labels). "
               "A mint ring = the name's recent correlation profile shifted > 2σ "
               "from its long-run pattern.")
    st.dataframe(nodes[["centrality", "cluster", "shift_z", "anomalous"]]
                 .sort_values("centrality", ascending=False).round(3),
                 width="stretch", height=280)


# ---------------------------------------------------------------- Stress Test

with tabs[4]:
    from src.models.stress import SCENARIOS

    table = load_stress()
    name = st.selectbox("Scenario", list(SCENARIOS),
                        format_func=lambda n: n.replace("_", " "))
    st.caption(SCENARIOS[name].description)

    base, shocked = table.loc["baseline"], table.loc[name]
    c1, c2, c3 = st.columns(3)
    kpi(c1, "VaR 95 (daily)", f"{shocked['var_95']:.2%}",
        f"baseline {base['var_95']:.2%}", "down")
    kpi(c2, "Ann. volatility", f"{shocked['ann_vol']:.1%}",
        f"baseline {base['ann_vol']:.1%}", "down")
    kpi(c3, "Max drawdown", f"{shocked['max_drawdown']:.1%}",
        f"baseline {base['max_drawdown']:.1%}", "down")

    st.write("")
    metrics = ["var_95", "ann_vol", "max_drawdown"]
    labels = ["VaR 95", "Ann. vol", "|Max drawdown|"]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[abs(base[m]) for m in metrics], name="baseline",
                marker_color=MUTED)
    fig.add_bar(x=labels, y=[abs(shocked[m]) for m in metrics],
                name=name.replace("_", " "), marker_color=MINT)
    fig.update_layout(barmode="group", yaxis_tickformat=".0%",
                      title="Before vs after — scenario damage")
    st.plotly_chart(themed(fig, 380), width="stretch")

    st.dataframe(
        table.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}",
                            "var_95": "{:.2%}", "max_drawdown": "{:.1%}"}),
        width="stretch")


# ---------------------------------------------------------------- Fundamentals

with tabs[5]:
    st.caption("Accounting lens — latest 10-K fundamentals straight from SEC "
               "EDGAR (XBRL). Prices say how *risky* each name is; the filings "
               "say how *healthy* it is.")
    fund = load_fundamentals()
    if fund is None:
        st.warning("Fundamentals unavailable — SEC fetch failed and no bundled "
                   "snapshot was found.")
    else:
        rat, dup = fund
        c = st.columns(4)
        kpi(c[0], "Companies", f"{len(rat)}", "with 10-K data", "flat")
        kpi(c[1], "Median ROE", f"{rat['roe'].median():.0%}",
            "return on equity", "flat")
        kpi(c[2], "Median net margin", f"{rat['net_margin'].median():.0%}",
            "profit per $ of sales", "flat")
        kpi(c[3], "Median D/E", f"{rat['debt_to_equity'].median():.2f}",
            "leverage", "flat")

        st.write("")
        left, right = st.columns(2, gap="large")

        # -- DuPont map: the two routes to ROE (margin vs asset velocity)
        with left:
            d = dup.dropna(subset=["net_margin", "asset_turnover", "roe"])
            fig = go.Figure()
            fig.add_scatter(
                x=d["asset_turnover"], y=d["net_margin"], mode="markers+text",
                text=d.index, textposition="top center",
                textfont=dict(size=10, color=TEXT),
                marker=dict(size=(d["roe"].clip(lower=0) * 34 + 10),
                            color=d["roe"], colorscale=[[0, MUTED], [1, MINT]],
                            showscale=True, colorbar=dict(title="ROE", tickformat=".0%"),
                            line=dict(width=1, color=BORDER)))
            fig.update_layout(
                title="DuPont map — margin vs asset turnover (bubble = ROE)",
                xaxis_title="asset turnover  (Sales / Assets)",
                yaxis_title="net margin", yaxis_tickformat=".0%")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Two routes to the same ROE: fat **margins** (top-left, e.g. "
                       "Apple/NVDA) vs asset **velocity** (bottom-right, e.g. Walmart).")

        # -- ROE league table
        with right:
            roe = rat["roe"].dropna().sort_values()
            fig = go.Figure()
            fig.add_bar(x=roe.values, y=roe.index, orientation="h",
                        marker_color=MINT)
            fig.update_layout(title="Return on equity by company (latest FY)",
                              xaxis_tickformat=".0%")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("ROE is the headline; the DuPont map on the left shows "
                       "*how* each name gets there.")

        st.write("")
        pct = "{:.1%}"
        st.dataframe(
            rat.style.format({
                "current_ratio": "{:.2f}", "quick_ratio": "{:.2f}",
                "debt_to_equity": "{:.2f}", "interest_coverage": "{:.1f}",
                "gross_margin": pct, "operating_margin": pct, "net_margin": pct,
                "roa": pct, "roe": pct, "asset_turnover": "{:.2f}"}),
            width="stretch")
        st.caption("Liquidity · leverage · profitability · efficiency. Blanks are "
                   "line items a filer doesn't report (e.g. banks have no current "
                   "ratio — no classified balance sheet).")


# ---------------------------------------------------------------- Forensic

with tabs[6]:
    st.caption("Forensic lens — the distress and earnings-manipulation screens "
               "auditors, credit desks and short-sellers run. Anomaly detection, "
               "but on the *financial statements* instead of the price.")
    fx = load_forensic()
    if fx is None:
        st.warning("Forensic scores unavailable — fundamentals could not be loaded.")
    else:
        summary, altman, beneish = fx["summary"], fx["altman"], fx["beneish"]
        bmad = fx["benford_mad"]

        distress = int((summary["altman_z"] < 1.81).sum())
        strong = int((summary["piotroski_f"] >= 7).sum())
        flags = int((summary["m_flag"] == True).sum())  # noqa: E712
        c = st.columns(4)
        kpi(c[0], "Altman distress", f"{distress}", "names with Z < 1.81",
            "down" if distress else "flat")
        kpi(c[1], "Piotroski strong", f"{strong}", "F-score ≥ 7", "flat")
        kpi(c[2], "Beneish flags", f"{flags}", "M > −1.78 (manipulation)",
            "down" if flags else "flat")
        kpi(c[3], "Benford", bmad["verdict"].split()[0].title(),
            f"MAD {bmad['mad']:.3f} · n={bmad['n']:,}", "flat")

        st.write("")
        left, right = st.columns(2, gap="large")

        # -- Altman Z with distress/grey/safe zone lines
        with left:
            z = altman.dropna(subset=["z_score"]).sort_values("z_score")
            zcolor = {"safe": MINT, "grey": AMBER, "distress": RED}
            # Market-value Z can run very high for megacaps (NVDA ≈ 62); clip the
            # bars so the 1.81/2.99 zone lines stay readable, but label true values.
            zplot = z["z_score"].clip(upper=13)
            fig = go.Figure()
            fig.add_bar(x=zplot, y=z.index, orientation="h",
                        marker_color=[zcolor.get(v, MUTED) for v in z["zone"]],
                        text=[f"{v:.1f}" for v in z["z_score"]], textposition="outside",
                        textfont=dict(color=TEXT, size=10))
            fig.add_vline(x=1.81, line=dict(color=RED, dash="dot", width=1),
                          annotation_text="distress")
            fig.add_vline(x=2.99, line=dict(color=MINT, dash="dot", width=1),
                          annotation_text="safe")
            fig.update_layout(title="Altman Z-score — distance from insolvency",
                              xaxis_range=[0, 15])
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Green safe · amber grey-zone · red distress. Banks and "
                       "regulated utilities are excluded (the model is calibrated "
                       "for operating companies).")

        # -- Beneish M with the manipulation threshold
        with right:
            m = beneish.dropna(subset=["m_score"]).sort_values("m_score")
            fig = go.Figure()
            fig.add_bar(x=m["m_score"], y=m.index, orientation="h",
                        marker_color=[RED if v else MUTED for v in m["manipulation_flag"]])
            fig.add_vline(x=-1.78, line=dict(color=RED, dash="dot", width=1),
                          annotation_text="−1.78 flag")
            fig.update_layout(title="Beneish M-score — earnings-manipulation screen")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Above −1.78 flags likely manipulation. Beneish penalises "
                       "rapid sales growth, so hyper-growth names (e.g. NVDA) can "
                       "trip it — a false positive worth reading, not a verdict.")

        # -- Benford's Law: do the reported digits look natural?
        st.write("")
        b = fx["benford"]
        fig = go.Figure()
        fig.add_bar(x=list(b.index), y=b["observed"], name="observed",
                    marker_color=MINT)
        fig.add_scatter(x=list(b.index), y=b["expected"], name="Benford expected",
                        mode="lines+markers", line=dict(color=AMBER, width=2))
        fig.update_layout(title="Benford's Law — first digit of every reported figure",
                          xaxis_title="leading digit", yaxis_tickformat=".0%",
                          xaxis=dict(tickmode="linear"))
        st.plotly_chart(themed(fig, 320), width="stretch")

        # -- Merton distance-to-default: the same distress read from the MARKET
        credit = load_credit()
        if credit is not None and credit["distance_to_default"].notna().any():
            st.write("")
            dtd = credit.dropna(subset=["distance_to_default"]).sort_values(
                "distance_to_default")
            # thinner cushion = redder; > ~4σ above the default point is
            # comfortably investment-grade, < 2σ is the danger zone.
            ddcolor = [RED if v < 2 else AMBER if v < 4 else MINT
                       for v in dtd["distance_to_default"]]
            fig = go.Figure()
            fig.add_bar(x=dtd["distance_to_default"], y=dtd.index, orientation="h",
                        marker_color=ddcolor,
                        text=[f"{v:.1f}σ" for v in dtd["distance_to_default"]],
                        textposition="outside", textfont=dict(color=TEXT, size=10))
            fig.add_vline(x=2, line=dict(color=RED, dash="dot", width=1),
                          annotation_text="thin")
            fig.update_layout(
                title="Merton distance-to-default — the market's credit view",
                xaxis_title="asset-vol standard deviations above the default point")
            st.plotly_chart(themed(fig, 320), width="stretch")
            st.caption("A structural (Merton 1974) model: equity is a call option "
                       "on the firm's assets, so equity value and volatility imply "
                       "how many standard deviations it sits above its debt. Altman "
                       "reads distress from the *accounting*; this reads the same "
                       "distress from the *market* — and banks drop out here too "
                       "(no current-liabilities tag for the default point).")

        st.write("")
        pct = "{:.1%}"
        st.dataframe(
            summary.style.format({"altman_z": "{:.2f}", "piotroski_f": "{:.0f}",
                                  "beneish_m": "{:.2f}", "accruals": pct}),
            width="stretch")
        st.caption("One row per name: bankruptcy risk (Altman), fundamental "
                   "quality 0–9 (Piotroski), manipulation odds (Beneish), and the "
                   "share of earnings not backed by cash (accruals). Blanks = a "
                   "model that doesn't apply to that filer (banks/utilities).")


# ---------------------------------------------------------------- Factors

with tabs[7]:
    st.caption("Factor lens — Fama-French 5 + momentum. The Deep Dive tab's CAPM "
               "uses one factor (the market); this uses six to separate *style "
               "exposure* from genuine *skill* (alpha that survives the factors).")
    fm = load_factor_model()
    if fm is None:
        st.warning("Factor data unavailable — Ken French library and snapshot "
                   "both unreachable.")
    else:
        model, loadings = fm
        FLABEL = {"mkt_rf": "Market", "smb": "Size", "hml": "Value",
                  "rmw": "Profit.", "cma": "Invest.", "mom": "Momentum"}
        order = list(FLABEL)

        c = st.columns(4)
        adir = "up" if model["alpha_ann"] > 0 else "down"
        kpi(c[0], "Factor alpha (ann.)", f"{model['alpha_ann']:+.1%}",
            f"t = {model['alpha_t']:.2f}", adir)
        kpi(c[1], "Market beta", f"{model['betas']['mkt_rf']:.2f}",
            "sensitivity to the market", "flat")
        kpi(c[2], "Model R²", f"{model['r2']:.2f}",
            f"{model['n']:,} trading days", "flat")
        nsig = sum(abs(model["tstats"][f]) > 1.96 for f in order)
        kpi(c[3], "Significant tilts", f"{nsig}/6", "|t| > 1.96", "flat")

        st.write("")
        left, right = st.columns(2, gap="large")

        # -- portfolio loadings (significant tilts in mint, rest muted)
        with left:
            betas = [model["betas"][f] for f in order]
            colors = [MINT if abs(model["tstats"][f]) > 1.96 else MUTED for f in order]
            fig = go.Figure()
            fig.add_bar(x=[FLABEL[f] for f in order], y=betas, marker_color=colors,
                        text=[f"t={model['tstats'][f]:.1f}" for f in order],
                        textposition="outside", textfont=dict(size=10, color=MUTED))
            fig.add_hline(y=0, line=dict(color=BORDER, width=1))
            fig.update_layout(title="Portfolio factor loadings (β)")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Mint = statistically significant (|t| > 1.96). This "
                       "portfolio is large-cap (negative Size) and quality-tilted, "
                       "with alpha left over after all six factors.")

        # -- per-name loadings heatmap
        with right:
            z = loadings[order]
            fig = go.Figure(go.Heatmap(
                z=z.values, x=[FLABEL[f] for f in order], y=list(z.index),
                colorscale=[[0, RED], [0.5, CARD], [1, MINT]], zmid=0,
                colorbar=dict(title="β")))
            fig.update_layout(title="Factor exposures by name")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Value (HML) lights up for JPM/XOM; NVDA carries momentum "
                       "and a growth (negative Value) tilt — textbook style exposures.")

        st.write("")
        st.dataframe(
            loadings.style.format({**{f: "{:+.2f}" for f in order},
                                   "alpha_ann": "{:+.1%}", "r2": "{:.2f}"}),
            width="stretch")
        st.caption("Per-name factor betas, annualized factor-adjusted alpha, and "
                   "R². Diversifying across names lifts the portfolio R² (0.90) "
                   "well above most single names — idiosyncratic noise nets out.")


# ---------------------------------------------------------------- Allocation

with tabs[8]:
    st.caption("Allocation lens — Markowitz optimisation. Everything else "
               "analyses the equal-weight book; this asks what the weights "
               "*should* be. Long-only, fully invested.")
    opt = load_optimization()
    if opt is None:
        st.warning("Optimisation unavailable — returns could not be loaded.")
    else:
        stats, weights, ef = opt["stats"], opt["weights"], opt["ef"]
        eq, mv = stats.loc["Equal-weight"], stats.loc["Min-variance"]
        ms, rp = stats.loc["Max-Sharpe"], stats.loc["Risk-parity"]

        c = st.columns(4)
        kpi(c[0], "Max-Sharpe", f"{ms['sharpe']:.2f}",
            f"vs 1/N {eq['sharpe']:.2f}", "up")
        kpi(c[1], "Min-variance vol", f"{mv['vol']:.1%}",
            f"vs 1/N {eq['vol']:.1%}", "up")  # lower risk = good = mint
        kpi(c[2], "1/N Sharpe", f"{eq['sharpe']:.2f}", "current book", "flat")
        kpi(c[3], "Risk-parity vol", f"{rp['vol']:.1%}",
            "equal risk contributions", "flat")

        st.write("")
        left, right = st.columns([3, 2], gap="large")

        # -- the efficient frontier with assets, the four portfolios, and the CML
        with left:
            pcolor = {"Equal-weight": MUTED, "Min-variance": AMBER,
                      "Max-Sharpe": MINT, "Risk-parity": "#58A6FF"}
            fig = go.Figure()
            fig.add_scatter(x=ef["vol"], y=ef["return"], mode="lines",
                            name="efficient frontier", line=dict(color=MINT, width=2))
            fig.add_scatter(x=opt["asset_vol"], y=opt["mu"], mode="markers+text",
                            text=opt["asset_vol"].index, textposition="bottom center",
                            marker=dict(size=6, color=MUTED), name="assets",
                            textfont=dict(size=9, color=MUTED))
            for name, row in stats.iterrows():
                fig.add_scatter(x=[row["vol"]], y=[row["return"]], mode="markers+text",
                                text=[name], textposition="top center",
                                marker=dict(size=13, color=pcolor[name],
                                            line=dict(width=1, color=TEXT)),
                                textfont=dict(size=10, color=TEXT), showlegend=False)
            # capital market line: from the risk-free rate through the tangency portfolio
            xmax = float(max(opt["asset_vol"].max(), ef["vol"].max())) * 1.05
            slope = (ms["return"] - opt["rf"]) / ms["vol"]
            fig.add_scatter(x=[0, xmax], y=[opt["rf"], opt["rf"] + slope * xmax],
                            mode="lines", name="Capital Market Line",
                            line=dict(color=AMBER, dash="dash", width=1))
            fig.update_layout(title="Efficient frontier — risk vs return",
                              xaxis_title="volatility (ann.)", yaxis_title="return (ann.)",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(themed(fig, 420), width="stretch")
            st.caption("Each optimal portfolio sits on the frontier; the dashed "
                       "Capital Market Line runs from the risk-free rate through "
                       "Max-Sharpe (the best risk-adjusted mix).")

        # -- optimal weights, side by side
        with right:
            fig = go.Figure(go.Heatmap(
                z=weights.values, x=list(weights.columns), y=list(weights.index),
                colorscale=[[0, CARD], [1, MINT]], zmin=0,
                colorbar=dict(title="weight", tickformat=".0%")))
            fig.update_layout(title="Optimal weights by portfolio")
            st.plotly_chart(themed(fig, 420), width="stretch")

        st.write("")
        st.dataframe(
            stats.style.format({"return": "{:.1%}", "vol": "{:.1%}", "sharpe": "{:.2f}"}),
            width="stretch")
        st.caption("Max-Sharpe lifts risk-adjusted return above the 1/N book by "
                   "concentrating in the best reward-per-unit-risk names; "
                   "Min-variance trades return for the lowest possible risk; "
                   "Risk-parity equalises each name's risk share (ties back to the "
                   "Euler decomposition in Deep Dive).")


# ---------------------------------------------------------------- ML Models

with tabs[9]:
    st.caption("Course-ML lens — the classical toolkit applied to the book. "
               "*Supervised* (predict a drawdown before it happens) and "
               "*unsupervised* (find the market's hidden structure), the "
               "complement to the deep-learning autoencoder in Anomalies.")

    # --- supervised: stress-day classifier ---------------------------------
    st.subheader("Supervised — will a drawdown follow?")
    clf = load_classifier()
    if clf is None:
        st.warning("Classifier unavailable — returns could not be loaded.")
    else:
        best_name = max(clf["models"], key=lambda m: clf["models"][m]["roc_auc"])
        best = clf["models"][best_name]
        c = st.columns(4)
        kpi(c[0], "Best model ROC-AUC", f"{best['roc_auc']:.2f}",
            f"{best_name.replace('_', ' ')}, out-of-sample", "up")
        kpi(c[1], "Time-series CV AUC", f"{best['cv_auc_mean']:.2f}",
            f"±{best['cv_auc_std']:.2f} across folds", "flat")
        kpi(c[2], "Stress-day base rate", f"{clf['positive_rate']:.0%}",
            f">{clf['threshold']:.0%} drop in {clf['horizon_days']}d", "flat")
        kpi(c[3], "Recall (caught)", f"{best['recall']:.0%}",
            "of real stress days flagged", "flat")

        st.write("")
        left, right = st.columns(2, gap="large")

        # ROC curves — both models vs the coin-flip diagonal
        with left:
            fig = go.Figure()
            palette = {"logistic": MINT, "random_forest": AMBER, "xgboost": "#58A6FF"}
            for name, m in clf["models"].items():
                fig.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                                mode="lines", name=f"{name} ({m['roc_auc']:.2f})",
                                line=dict(color=palette.get(name, MUTED), width=2))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                            line=dict(color=BORDER, dash="dash", width=1))
            fig.update_layout(title="ROC curve (out-of-sample)",
                              xaxis_title="false positive rate",
                              yaxis_title="true positive rate")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Above the diagonal = better than chance. The label is "
                       "forward-looking and the split is chronological, so this is "
                       "an honest out-of-sample read — no peeking at the future.")

        # feature importance — what precedes drawdowns
        with right:
            imp = dict(sorted(clf["feature_importance"].items(),
                              key=lambda kv: kv[1]))
            fig = go.Figure(go.Bar(
                x=list(imp.values()), y=list(imp), orientation="h",
                marker_color=MINT))
            fig.update_layout(title="Random-forest feature importance")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("Volatility (rolling and EWMA) dominates — turbulence "
                       "clusters, so a rough market is the strongest tell that a "
                       "drawdown is coming.")

    st.divider()

    # --- unsupervised: PCA + KMeans ----------------------------------------
    st.subheader("Unsupervised — the market's hidden structure")
    cl = load_clusters()
    if cl is None:
        st.warning("PCA/clustering unavailable — returns could not be loaded.")
    else:
        pca, km = cl["pca"], cl["km"]
        c = st.columns(3)
        kpi(c[0], "PC1 variance", f"{pca['explained_variance_ratio'][0]:.0%}",
            "one common 'market' factor", "flat")
        kpi(c[1], "Top-3 PCs", f"{pca['cumulative_variance'][2]:.0%}",
            "of all co-movement", "flat")
        kpi(c[2], "KMeans clusters", f"k = {km['k']}",
            f"silhouette {km['silhouette']:.2f}", "flat")

        st.write("")
        left, right = st.columns(2, gap="large")

        # PCA scree — variance explained per component
        with left:
            evr = pca["explained_variance_ratio"]
            labels = [f"PC{i + 1}" for i in range(len(evr))]
            fig = go.Figure()
            fig.add_bar(x=labels, y=evr, marker_color=MINT, name="individual")
            fig.add_scatter(x=labels, y=pca["cumulative_variance"], mode="lines+markers",
                            name="cumulative", line=dict(color=AMBER, width=2))
            fig.update_layout(title="PCA scree — variance explained",
                              yaxis_tickformat=".0%")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("PC1 is the market itself (every name loads the same sign); "
                       "PC2 onward are style rotations. The *statistical* mirror of "
                       "the *economic* Fama-French factors in the Factors tab.")

        # KMeans clusters on the PC1/PC2 plane
        with right:
            load = pca["loadings"]
            palette = [MINT, AMBER, "#58A6FF", RED, MUTED]
            fig = go.Figure()
            for cid in sorted(set(km["labels"].values())):
                names = [t for t, c_ in km["labels"].items() if c_ == cid]
                fig.add_scatter(
                    x=load.loc[names, "PC1"], y=load.loc[names, "PC2"],
                    mode="markers+text", text=names, textposition="top center",
                    marker=dict(size=12, color=palette[cid % len(palette)],
                                line=dict(width=1, color=TEXT)),
                    textfont=dict(size=10, color=TEXT), name=f"cluster {cid}")
            fig.update_layout(title="KMeans clusters on the PC1/PC2 plane",
                              xaxis_title="PC1 loading", yaxis_title="PC2 loading")
            st.plotly_chart(themed(fig, 380), width="stretch")
            st.caption("KMeans clusters the names by their correlation profile and "
                       "recovers the same sector blocks the correlation network "
                       "finds — two unsupervised methods agreeing is the signal.")


# ---------------------------------------------------------------- Ask the Agent

with tabs[10]:
    from src.agent import memo as agent

    llm_ready = agent._client() is not None
    if not llm_ready:
        st.info("No ANTHROPIC_API_KEY configured — the agent runs in template "
                "mode. Q&A is disabled; the memo below uses the deterministic "
                "fallback filled with real computed numbers.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Ask a question")
        question = st.text_area(
            "Question", placeholder="e.g. Which scenario hurts the portfolio "
            "most, and which names drive it?", label_visibility="collapsed")
        if st.button("Ask", type="primary", disabled=not llm_ready):
            with st.spinner("The agent is calling the risk engine…"):
                st.session_state["answer"] = agent.ask(question)
        if "answer" in st.session_state:
            st.markdown(st.session_state["answer"])

    with right:
        st.subheader("Risk memo")
        if st.button("Generate Risk Memo", type="primary"):
            with st.spinner("Writing memo…"):
                st.session_state["memo"] = agent.write_memo()
        if "memo" in st.session_state:
            from src.agent.report import save_markdown, save_pdf
            memo_text = st.session_state["memo"]
            pdf_path = save_pdf(memo_text)
            save_markdown(memo_text)
            st.download_button("Download PDF", data=pdf_path.read_bytes(),
                               file_name="sentinel_risk_memo.pdf",
                               mime="application/pdf")
            st.markdown(memo_text)
