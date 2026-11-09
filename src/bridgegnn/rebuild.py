"""Time-respecting rebuild of the comorbidity graph from development records before a cutoff (design 5a, B5).

Reuses Project 1's own functions so the pre-cutoff graph is built by the same rules as the main graph:
Fisher + Bonferroni + OR + support edges, phi weights, Leiden at the P1 resolution, 100-seed stable cores,
cluster-restricted betweenness bridges. The node set is fixed to the 496 main-graph nodes so node features align.
"""
from __future__ import annotations

import datetime as dt

import networkx as nx
import numpy as np
import polars as pl
from bridgedx import centrality as p1c
from bridgedx import communities as p1cm
from bridgedx import edges as p1e
from bridgedx import graph as p1g
from sklearn.metrics import adjusted_rand_score

from .nodefeat import STRUCT_COLS

TIME_INVARIANT = ["inpatient_share", "ccir_chronic_share", "mean_age_first_occurrence", "female_share", "name", "chapter"]


def _check_table(pc: pl.DataFrame, test_ids: set, cutoff: str) -> None:
    assert not (set(pc["person_id"].to_list()) & set(test_ids)), "test patients in the rebuild table"
    assert (pc["first_date"] < dt.date.fromisoformat(cutoff)).all(), "records on/after the cutoff in the rebuild table"


def restricted_patient_category(codes_long: pl.DataFrame, split: pl.DataFrame, dev_splits: list[str], era_start: str,
                                cutoff: str) -> pl.DataFrame:
    """(person_id, category, first_date) over development patients' records in [era_start, cutoff)."""
    dev_ids = split.filter(pl.col("split").is_in(dev_splits))["person_id"].to_list()
    test_ids = set(split.filter(pl.col("split") == "test")["person_id"].to_list())
    pc = (codes_long.filter(pl.col("person_id").is_in(dev_ids)
                            & (pl.col("condition_date") >= dt.date.fromisoformat(era_start))
                            & (pl.col("condition_date") < dt.date.fromisoformat(cutoff)))
          .group_by("person_id", "category").agg(pl.col("condition_date").min().alias("first_date"))
          .sort("person_id", "category"))
    _check_table(pc, test_ids, cutoff)
    return pc


def build_graph(pc: pl.DataFrame, nodes: pl.DataFrame, p1_cfg: dict) -> tuple[nx.Graph, pl.DataFrame]:
    """Edges by the P1 rule on the restricted table; node set = `nodes` (prevalence recomputed on the table)."""
    cats = nodes["category"].to_list()
    n_pat = pc["person_id"].n_unique()
    prev = pc.filter(pl.col("category").is_in(cats)).group_by("category").agg(pl.len().alias("n_patients"))
    node_tab = (nodes.select("category", *[c for c in ["name", "chapter", "flagged"] if c in nodes.columns])
                .join(prev, on="category", how="left").with_columns(pl.col("n_patients").fill_null(0))
                .with_columns((pl.col("n_patients") / max(n_pat, 1)).alias("prevalence")))
    pairs = p1e.pair_counts(pc, cats)
    pairs = p1e.weights(p1e.fisher_filter(pairs, p1_cfg))
    g = p1g.build(node_tab, pairs, weight="phi")
    return g, pairs


def cores_and_bridges(g: nx.Graph, p1_cfg: dict, anchors: dict[str, list[str]], n_seeds: int | None = None,
                      n_null: int | None = None) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    partition = p1cm.leiden(g, p1_cfg, seed=int(p1_cfg["communities"]["seeds"][0]))
    co = p1cm.coassignment(g, p1_cfg, n_seeds)
    min_share = float(p1_cfg["centrality"].get("core_min_coassignment", 0.9))
    cores = {t: set(p1cm.stable_core(co, a, min_share)) for t, a in anchors.items()}
    rows = [{"target": t, "category": c, "in_core": c in cores[t]} for t in cores for c in g.nodes]
    target_defs = pl.DataFrame(rows)
    clusters = {t: s for t, s in cores.items() if s}
    bridges = p1c.cluster_bridges(g, clusters, p1_cfg, n_null=n_null, seed=int(p1_cfg["communities"]["seeds"][0]))
    return partition, target_defs, bridges


def node_features(g: nx.Graph, partition: dict, bridges: pl.DataFrame, p1_node_features: pl.DataFrame) -> pl.DataFrame:
    """Structural columns recomputed on g; time-invariant columns copied from the P1 table (stated in the report)."""
    nodes = list(g.nodes)
    bt = p1c.weighted_betweenness(g)
    deg = dict(g.degree())
    wdeg = dict(g.degree(weight="weight"))
    oth = p1c.other_centralities(g)
    pr = nx.pagerank(g, weight="weight")
    roles = p1c.roles(g, partition)
    try:
        resid = p1c.prevalence_residual(g)
    except (np.linalg.LinAlgError, ValueError):   # degenerate when every prevalence is 0 (synthetic graphs)
        resid = pl.DataFrame({"category": nodes})
    from bridgedx.viz import sparsify_top_k
    bb = sparsify_top_k(g, 10)
    base = pl.DataFrame({"category": nodes,
                         "prevalence": [float(g.nodes[v].get("prevalence", 0.0)) for v in nodes],
                         "degree": [int(deg[v]) for v in nodes], "weighted_degree": [float(wdeg[v]) for v in nodes],
                         "betweenness": [float(bt[v]) for v in nodes], "pagerank": [float(pr[v]) for v in nodes],
                         "backbone_degree": [int(bb.degree(v)) if v in bb else 0 for v in nodes],
                         "community_seed0": [int(partition[v]) for v in nodes]})
    base = base.join(oth, on="category", how="left")
    role_cols = [c for c in ["participation", "within_module_z"] if c in roles.columns]
    base = base.join(roles.select("category", *role_cols), on="category", how="left")
    if "degree_residual_z" in resid.columns:
        base = base.join(resid.select("category", "degree_residual_z"), on="category", how="left")
    # null percentile of betweenness is not recomputed (100 rewirings); carry P1's as a documented approximation
    inv = [c for c in TIME_INVARIANT + ["bt_null_percentile"] if c in p1_node_features.columns]
    base = base.join(p1_node_features.select("category", *inv), on="category", how="left")
    surv = set(bridges.filter(pl.col("bridge_pass"))["category"].to_list()) if "bridge_pass" in bridges.columns else set()
    base = base.with_columns(pl.col("category").is_in(list(surv)).alias("bridge_survivor"))
    for c in STRUCT_COLS:
        if c not in base.columns:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return base.sort("category")


def compare(g_new: nx.Graph, g_u: nx.Graph, part_new: dict, part_u: dict, bridges_new: pl.DataFrame, bridges_u: pl.DataFrame) -> dict:
    e_new = {frozenset(e) for e in g_new.edges()}
    e_u = {frozenset(e) for e in g_u.edges()}
    common = [v for v in g_u.nodes if v in part_new and v in part_u]
    b_new = set(bridges_new.filter(pl.col("bridge_pass"))["category"].to_list())
    b_u = set(bridges_u.filter(pl.col("bridge_pass"))["category"].to_list())
    return {"n_edges_new": len(e_new), "n_edges_u": len(e_u),
            "edge_jaccard": len(e_new & e_u) / max(len(e_new | e_u), 1),
            "ari": float(adjusted_rand_score([part_u[v] for v in common], [part_new[v] for v in common])) if common else float("nan"),
            "bridge_jaccard": len(b_new & b_u) / max(len(b_new | b_u), 1), "n_bridges_new": len(b_new), "n_bridges_u": len(b_u),
            "n_isolates_new": sum(1 for _, d in g_new.degree() if d == 0)}
