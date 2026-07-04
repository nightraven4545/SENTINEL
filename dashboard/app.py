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
tabs = st.tabs(["Overview", "Anomalies", "Network", "Stress Test", "Ask the Agent"])


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
    st.plotly_chart(themed(fig), use_container_width=True)

    roll = risk.rolling_vol(port)
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name="21d rolling vol",
                    line=dict(width=1.5, color=MINT))
    fig.add_hline(y=full_vol, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text="full-sample")
    fig.update_layout(title="Portfolio rolling volatility (annualized)",
                      yaxis_tickformat=".0%")
    st.plotly_chart(themed(fig, 320), use_container_width=True)


# ---------------------------------------------------------------- Anomalies

with tabs[1]:
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
    st.plotly_chart(themed(fig), use_container_width=True)

    st.caption("IsolationForest + PyTorch autoencoder over per-name returns and "
               "rolling vol. Mint = both models agree (high conviction).")
    flagged = anoms[anoms["if_flag"] | anoms["ae_flag"]].sort_index(ascending=False)
    st.dataframe(flagged.round(4), use_container_width=True, height=320)


# ---------------------------------------------------------------- Network

with tabs[2]:
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
    st.plotly_chart(themed(fig, 560), use_container_width=True)

    st.caption("Communities are discovered from returns alone (no sector labels). "
               "A mint ring = the name's recent correlation profile shifted > 2σ "
               "from its long-run pattern.")
    st.dataframe(nodes[["centrality", "cluster", "shift_z", "anomalous"]]
                 .sort_values("centrality", ascending=False).round(3),
                 use_container_width=True, height=280)


# ---------------------------------------------------------------- Stress Test

with tabs[3]:
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
    st.plotly_chart(themed(fig, 380), use_container_width=True)

    st.dataframe(
        table.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}",
                            "var_95": "{:.2%}", "max_drawdown": "{:.1%}"}),
        use_container_width=True)


# ---------------------------------------------------------------- Ask the Agent

with tabs[4]:
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
