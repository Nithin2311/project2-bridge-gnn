"""Time-respecting rebuild of the comorbidity graph from development records before a cutoff (design 5a, B5).

Reuses Project 1's own functions so the pre-cutoff graph is built by the same rules as the main graph:
Fisher + Bonferroni over the SAME family (every pair among all candidate categories), OR and support floors, phi
weights, Leiden at the P1 resolution, 100-seed stable cores, cluster-restricted betweenness bridges. The node set is
fixed to the 496 main-graph nodes so node features align. Every node feature that depends on records is recomputed
on the restricted records; only the HCUP chronic share (an external lookup), names and chapters are copied.
"""
from __future__ import annotations

import datetime as dt
import warnings

import networkx as nx
import numpy as np
import polars as pl
from bridgedx import centrality as p1c
from bridgedx import communities as p1cm
from bridgedx import edges as p1e
from bridgedx import graph as p1g
from bridgedx.viz import sparsify_top_k
from sklearn.metrics import adjusted_rand_score

from .nodefeat import STRUCT_COLS

COPIED_FROM_P1 = ["ccir_chronic_share", "name", "chapter"]   # the only columns not recomputed on the restricted records


def _check_table(pc: pl.DataFrame, test_ids: set, cutoff: str) -> None:
    if set(pc["person_id"].to_list()) & set(test_ids):
        raise ValueError("test patients in the rebuild table")
    if not (pc["first_date"] < dt.date.fromisoformat(cutoff)).all():
        raise ValueError("records on/after the cutoff in the rebuild table")


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


def build_graph(pc: pl.DataFrame, nodes: pl.DataFrame, p1_cfg: dict, all_categories: list[str] | None = None,
                test_ids: set | None = None, cutoff: str | None = None) -> tuple[nx.Graph, pl.DataFrame]:
    """Edges by the P1 rule on the restricted table. The Bonferroni family is every pair among `all_categories`
    (P1's 617 candidate categories), as in P1; the graph keeps only `nodes` (the 496 main-graph nodes) and
    `graph.build` drops edges outside them. Prevalence is recomputed on the table."""
    if test_ids is not None and cutoff is not None:
        _check_table(pc, test_ids, cutoff)
    cats = nodes["category"].to_list()
    family = sorted(set(all_categories) | set(cats)) if all_categories is not None else cats
    n_pat = pc["person_id"].n_unique()
    prev = pc.filter(pl.col("category").is_in(cats)).group_by("category").agg(pl.len().alias("n_patients"))
    node_tab = (nodes.select("category", *[c for c in ["name", "chapter", "flagged"] if c in nodes.columns])
                .join(prev, on="category", how="left").with_columns(pl.col("n_patients").fill_null(0))
                .with_columns((pl.col("n_patients") / max(n_pat, 1)).alias("prevalence")))
    pairs = p1e.pair_counts(pc, family)
    pairs = p1e.weights(p1e.fisher_filter(pairs, p1_cfg))
    g = p1g.build(node_tab, pairs, weight="phi")
    return g, pairs


def cores_and_bridges(g: nx.Graph, p1_cfg: dict, anchors: dict[str, list[str]], n_seeds: int | None = None,
                      n_null: int | None = None) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Leiden partition (seed 0), target definitions with anchor co-assignment (as P1 writes them), bridges."""
    partition = p1cm.leiden(g, p1_cfg, seed=int(p1_cfg["communities"]["seeds"][0]))
    co = p1cm.coassignment(g, p1_cfg, n_seeds)
    min_share = float(p1_cfg["centrality"].get("core_min_coassignment", 0.9))
    cats = co["category"].to_list()
    M = co.drop("category").to_numpy()
    idx = {v: i for i, v in enumerate(cats)}
    rows = []
    cores = {}
    for t, a in anchors.items():
        present = [x for x in a if x in idx]
        core = set(p1cm.stable_core(co, a, min_share)) if present else set()
        cores[t] = core
        if not core:
            warnings.warn(f"target {t}: no stable core at co-assignment >= {min_share} on this graph (anchors {a}); "
                          "its bridge features will be zero", UserWarning, stacklevel=2)
        for v in g.nodes:
            per = [float(M[idx[x], idx[v]]) for x in present] if v in idx else []
            rows.append({"target": t, "category": v, "in_core": v in core,
                         "coassignment_mean_anchors": float(np.mean(per)) if per else None,
                         "coassignment_min_anchors": float(np.min(per)) if per else None,
                         **{f"coassignment_{x}": float(M[idx[x], idx[v]]) if v in idx else None for x in present}})
    target_defs = pl.DataFrame(rows)
    clusters = {t: s for t, s in cores.items() if s}
    empty = pl.DataFrame(schema={"pair": pl.Utf8, "category": pl.Utf8, "bridge_betweenness": pl.Float64, "bridge_pass": pl.Boolean})
    if len(clusters) < 2:
        warnings.warn("fewer than two non-empty cores: no bridge pairs can be screened", UserWarning, stacklevel=2)
        return partition, target_defs, empty
    bridges = p1c.cluster_bridges(g, clusters, p1_cfg, n_null=n_null, seed=int(p1_cfg["communities"]["seeds"][0]))
    if "pair" not in bridges.columns:
        bridges = empty
    return partition, target_defs, bridges


def node_features(g: nx.Graph, partition: dict, bridges: pl.DataFrame, p1_node_features: pl.DataFrame,
                  pc: pl.DataFrame | None = None, demo: pl.DataFrame | None = None, inpatient: pl.DataFrame | None = None,
                  n_null_bt: int | None = None, p1_cfg: dict | None = None, backbone_k: int = 10) -> pl.DataFrame:
    """Every structural and demographic column recomputed on g / the restricted records:
    prevalence, degree, weighted_degree, betweenness (+ null percentile from rewired graphs), closeness,
    eigenvector, pagerank, participation, within_module_z, degree_residual_z, backbone_degree, community_seed0,
    bridge_survivor, female_share and mean_age_first_occurrence (from pc + demo), inpatient_share (caller-supplied,
    computed on restricted records). Copied from P1: ccir_chronic_share, name, chapter."""
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
    bb = sparsify_top_k(g, backbone_k)
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
    cfg_bt = p1_cfg or {"communities": {"null_model": {"n": 100}, "seeds": [0]}}
    if n_null_bt is not None:
        cfg_bt = {"communities": {**cfg_bt["communities"], "null_model": {**cfg_bt["communities"]["null_model"], "n": int(n_null_bt)}}}
    nulls = p1c.betweenness_null_percentile(g, cfg_bt, observed=bt)
    base = base.with_columns(pl.Series("bt_null_percentile", [float(nulls[v]["bt_null_percentile"]) for v in nodes]))
    if pc is not None and demo is not None:
        d = (pc.join(demo.select("person_id", "sex", "birth_date"), on="person_id", how="inner")
             .with_columns(((pl.col("first_date") - pl.col("birth_date")).dt.total_days() / 365.25).alias("age"))
             .group_by("category").agg(pl.col("age").mean().alias("mean_age_first_occurrence"),
                                       (pl.col("sex") == "female").mean().alias("female_share")))
        base = base.join(d, on="category", how="left")
    if inpatient is not None:
        base = base.join(inpatient.select("category", "inpatient_share"), on="category", how="left")
    inv = [c for c in COPIED_FROM_P1 if c in p1_node_features.columns]
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
    non_iso = [v for v in common if g_new.degree(v) > 0 and g_u.degree(v) > 0]
    pairs_new = set(bridges_new["pair"].to_list()) if "pair" in bridges_new.columns else set()
    pairs_u = set(bridges_u["pair"].to_list()) if "pair" in bridges_u.columns else set()
    shared = pairs_new & pairs_u
    b_new = set(bridges_new.filter(pl.col("bridge_pass") & pl.col("pair").is_in(list(shared)))["category"].to_list()) if shared else set()
    b_u = set(bridges_u.filter(pl.col("bridge_pass") & pl.col("pair").is_in(list(shared)))["category"].to_list()) if shared else set()
    return {"n_edges_new": len(e_new), "n_edges_u": len(e_u),
            "edge_jaccard": len(e_new & e_u) / max(len(e_new | e_u), 1),
            "ari": float(adjusted_rand_score([part_u[v] for v in common], [part_new[v] for v in common])) if common else float("nan"),
            "ari_non_isolates": float(adjusted_rand_score([part_u[v] for v in non_iso], [part_new[v] for v in non_iso])) if non_iso else float("nan"),
            "n_shared_pairs": len(shared), "n_pairs_new": len(pairs_new), "n_pairs_u": len(pairs_u),
            "bridge_jaccard_shared_pairs": len(b_new & b_u) / max(len(b_new | b_u), 1) if shared else float("nan"),
            "n_bridges_new": int(bridges_new.filter(pl.col("bridge_pass"))["category"].n_unique()) if "bridge_pass" in bridges_new.columns else 0,
            "n_bridges_u": int(bridges_u.filter(pl.col("bridge_pass"))["category"].n_unique()) if "bridge_pass" in bridges_u.columns else 0,
            "n_isolates_new": sum(1 for _, d in g_new.degree() if d == 0)}
