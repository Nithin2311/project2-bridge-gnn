"""Graph layers for the experiments: the BH-filtered edge variant (U-bh), null graphs (Layer N), stats."""
from __future__ import annotations

import random

import networkx as nx
import polars as pl
from bridgedx.null_models import rewired
from statsmodels.stats.multitest import multipletests


def bh_edges(edges_all: pl.DataFrame, node_set: set[str], cfg: dict) -> tuple[pl.DataFrame, int]:
    """U-bh: Project 1's full pair table with BH-FDR on p_fisher in place of Bonferroni.

    BH runs over the SAME family Project 1 corrected over (every tested pair in edges.csv); the OR and support
    floors (from the P1 config, passed as cfg["p1_edges"]) and the main-graph node restriction are applied
    afterwards, exactly as in Project 1. Returns (edges, family size)."""
    family_n = edges_all.height
    q = multipletests(edges_all["p_fisher"].to_numpy(), method="fdr_bh")[1]
    nodes = list(node_set)
    floors = cfg.get("p1_edges", {"min_odds_ratio": 1.2, "min_shared_patients": 10})
    out = (edges_all.with_columns(pl.Series("q_bh", q))
           .filter(pl.col("u").is_in(nodes) & pl.col("v").is_in(nodes)
                   & (pl.col("odds_ratio") >= float(floors["min_odds_ratio"]))
                   & (pl.col("n_both") >= int(floors["min_shared_patients"]))
                   & (pl.col("q_bh") < float(cfg["layers"]["bh_alpha"]))))
    return out, family_n


def graph_from_edges(edges: pl.DataFrame, g_template: nx.Graph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(g_template.nodes(data=True))
    for u, v, phi in zip(edges["u"], edges["v"], edges["phi"]):
        g.add_edge(u, v, weight=float(phi), phi=float(phi), distance=1.0 / float(phi))
    g.graph.update(g_template.graph)
    return g


def rewired_graphs(g_u: nx.Graph, n: int, seed: int) -> list[nx.Graph]:
    return list(rewired(g_u, n, seed=seed))


def shuffled_graph(g_u: nx.Graph, seed: int) -> nx.Graph:
    """Same nodes and edge count, endpoints drawn uniformly (no loops, no duplicates), weights reassigned at
    random. Does NOT preserve degrees (that is the point: it is the second null beside the rewired one)."""
    rng = random.Random(seed)
    nodes = list(g_u.nodes)
    m = g_u.number_of_edges()
    weights = [d["weight"] for _, _, d in g_u.edges(data=True)]
    rng.shuffle(weights)
    h = nx.Graph()
    h.add_nodes_from(g_u.nodes(data=True))
    seen = set()
    while len(seen) < m:
        u, v = rng.sample(nodes, 2)
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen.add(key)
    for (u, v), w in zip(sorted(seen), weights):
        h.add_edge(u, v, weight=w, phi=w, distance=1.0 / w)
    h.graph.update(g_u.graph)
    return h


def layer_stats(g: nx.Graph) -> dict:
    n, m = g.number_of_nodes(), g.number_of_edges()
    return {"n_nodes": n, "n_edges": m, "density": nx.density(g), "n_isolates": sum(1 for _, d in g.degree() if d == 0),
            "mean_degree": (2 * m / n) if n else 0.0}
