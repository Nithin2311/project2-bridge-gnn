"""Graph layers for the experiments: the BH-filtered edge variant (U-bh), null graphs (Layer N), stats."""
from __future__ import annotations

import random

import networkx as nx
import polars as pl
from statsmodels.stats.multitest import multipletests

from bridgedx.null_models import rewired

OR_FLOOR = 1.2        # same floors as Project 1 config edges.min_odds_ratio / min_shared_patients
SUPPORT_FLOOR = 10


def bh_edges(edges_all: pl.DataFrame, node_set: set[str], cfg: dict) -> pl.DataFrame:
    """Project 1's full pair table filtered by BH-FDR on p_fisher instead of Bonferroni; same OR and support floors."""
    nodes = list(node_set)
    cand = edges_all.filter(pl.col("u").is_in(nodes) & pl.col("v").is_in(nodes)
                            & (pl.col("odds_ratio") >= OR_FLOOR) & (pl.col("n_both") >= SUPPORT_FLOOR))
    if cand.height == 0:
        return cand.with_columns(pl.lit(None, dtype=pl.Float64).alias("q_bh"))
    q = multipletests(cand["p_fisher"].to_numpy(), method="fdr_bh")[1]
    return cand.with_columns(pl.Series("q_bh", q)).filter(pl.col("q_bh") < float(cfg["layers"]["bh_alpha"]))


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
    """Same nodes and edge count, endpoints drawn uniformly (no loops, no duplicates), weights reassigned at random."""
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
