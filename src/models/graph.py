"""Correlation network of the portfolio (NetworkX).

Assets are nodes; an edge connects two assets whose return correlation
exceeds a threshold. On top of that graph we compute:

- eigenvector centrality — "systemic importance": a name is central when it
  co-moves with names that are themselves central. High-centrality assets
  are where contagion spreads from.
- communities — clusters of names that move together (should recover
  sectors without being told about sectors).
- correlation-shift anomalies — assets whose RECENT correlation profile
  looks unlike their long-run one. A defensive name suddenly trading like
  tech is a risk signal even if its own vol is unchanged.

graph_payload() returns a plain dict (nodes with layout x/y, edges with
weights) so the dashboard can render it without importing networkx.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

# why 0.45: below the tight tech-block correlations (~0.6+) but above the
# defensive/energy cross-pairs (~0.3) — keeps the graph readable instead of
# fully connected. Tune per portfolio via the parameter.
EDGE_THRESHOLD = 0.45

# why 63: ~one quarter of trading days — long enough for a stable correlation
# estimate, short enough to catch a regime change while it still matters.
SHIFT_WINDOW = 63

# why 2.0: the usual "two sigma" flag; with ~10 assets expect zero or one
# flagged name in calm markets.
SHIFT_Z_FLAG = 2.0

LAYOUT_SEED = 42  # deterministic node positions across runs


def build_graph(corr: pd.DataFrame, threshold: float = EDGE_THRESHOLD) -> nx.Graph:
    """Graph with every asset as a node and edges where correlation > threshold."""
    g = nx.Graph()
    g.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            weight = float(corr.loc[a, b])
            if weight > threshold:
                g.add_edge(a, b, weight=weight)
    return g


def centrality(g: nx.Graph) -> pd.Series:
    """Weighted eigenvector centrality, computed per connected component.

    Eigenvector centrality is undefined across disconnected graphs (and ours
    can disconnect: defensives drop out at the edge threshold). Solving each
    component separately keeps rankings meaningful within groups; isolated
    nodes get 0 — no connections, no systemic importance.
    """
    scores: dict[str, float] = {node: 0.0 for node in g.nodes}
    for component in nx.connected_components(g):
        if len(component) > 1:
            sub = g.subgraph(component)
            scores.update(nx.eigenvector_centrality_numpy(sub, weight="weight"))
    return pd.Series(scores, name="centrality")


def communities(g: nx.Graph) -> dict[str, int]:
    """Greedy-modularity clusters, as {ticker: cluster_id}."""
    groups = nx.community.greedy_modularity_communities(g, weight="weight")
    return {node: i for i, group in enumerate(groups) for node in group}


def correlation_shift(returns: pd.DataFrame, window: int = SHIFT_WINDOW) -> pd.DataFrame:
    """How far each asset's recent correlation profile drifted from its
    long-run profile.

    For each asset: mean absolute difference between its correlation row
    computed on the last `window` days vs the full sample, z-scored across
    assets. Flag = this name's relationships changed abnormally vs its peers'.
    """
    full = returns.corr()
    recent = returns.tail(window).corr()
    # mean |Δcorr| per asset; diagonal is 0-0 so it dilutes uniformly — fine.
    shift = (recent - full).abs().mean()
    z = (shift - shift.mean()) / shift.std()
    return pd.DataFrame({"shift": shift, "zscore": z, "flag": z > SHIFT_Z_FLAG})


def graph_payload(returns: pd.DataFrame,
                  threshold: float = EDGE_THRESHOLD,
                  window: int = SHIFT_WINDOW) -> dict:
    """Everything the dashboard needs to draw the network, as plain data."""
    corr = returns.corr()
    g = build_graph(corr, threshold)
    cent = centrality(g)
    clusters = communities(g)
    shifts = correlation_shift(returns, window)
    pos = nx.spring_layout(g, seed=LAYOUT_SEED, weight="weight")

    nodes = [
        {
            "id": t,
            "centrality": round(float(cent[t]), 4),
            "cluster": clusters.get(t, -1),          # -1 = isolated node
            "shift_z": round(float(shifts.loc[t, "zscore"]), 2),
            "anomalous": bool(shifts.loc[t, "flag"]),
            "x": round(float(pos[t][0]), 4),
            "y": round(float(pos[t][1]), 4),
        }
        for t in corr.columns
    ]
    edges = [
        {"source": a, "target": b, "weight": round(float(d["weight"]), 3)}
        for a, b, d in g.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges,
            "threshold": threshold, "window": window}


if __name__ == "__main__":
    from src.warehouse.duck import returns_wide

    payload = graph_payload(returns_wide())
    print(f"{len(payload['nodes'])} nodes, {len(payload['edges'])} edges")
    print(pd.DataFrame(payload["nodes"]).set_index("id")
          .sort_values("centrality", ascending=False))
