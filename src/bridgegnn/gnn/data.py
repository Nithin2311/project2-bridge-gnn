"""Per-row patient subgraphs as plain numpy Examples (no torch here; see models.to_pyg).

A row's node set = its codes in the lookback window that are main-graph nodes, not masked for the target and
not in `drop_codes`; optionally plus 1-hop layer neighbours as context nodes. Node features = selected static
columns of the node matrix, optional per-row temporal columns, and two flags (is_virtual, is_context). Edges =
the induced subgraph on the chosen layer. A row with no nodes left gets one virtual node.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import polars as pl

FEATURE_GROUPS = ["CODE", "DEMO", "UTIL", "TEMP", "STRUCT", "BRIDGE"]
STATIC_FOR = {"STRUCT": ["STRUCT", "CHAPTER", "COMMUNITY"], "BRIDGE": ["BRIDGE"]}
DEMO_NAMES = ["demo_age", "demo_female"]
UTIL_NAMES = ["util_n_codes_lb365", "util_n_codes_lb1095", "util_n_codes_lball", "util_n_rows_lb1095",
              "util_n_visit_days_365", "util_days_since_first_dx", "util_index_year"]
UNSCALED = {"demo_female"}
N_TEMP = 3
N_FLAGS = 2


@dataclass
class Options:
    layer: str
    features: list[str]
    lookback: int
    expand: bool
    weighted: bool
    drop_codes: set[str]
    directed: bool
    symmetrise: bool


@dataclass
class Example:
    x: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    node_ids: np.ndarray
    cov: np.ndarray
    y: int
    person_id: int
    index_date: dt.date
    split: str


def covariate_names(features: list[str]) -> list[str]:
    names: list[str] = []
    if "DEMO" in features:
        names += DEMO_NAMES
    if "UTIL" in features:
        names += UTIL_NAMES
    return names


def static_columns(groups: dict[str, list[str]], features: list[str]) -> list[int]:
    """Indices into X (column order = STRUCT, BRIDGE, CHAPTER, COMMUNITY as written by nodefeat)."""
    order = ["STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY"]
    offsets, off = {}, 0
    for gname in order:
        offsets[gname] = off
        off += len(groups.get(gname, []))
    wanted = []
    for f in features:
        for gname in STATIC_FOR.get(f, []):
            wanted += list(range(offsets[gname], offsets[gname] + len(groups.get(gname, []))))
    return wanted


def feature_columns(groups: dict, features: list[str]) -> dict:
    static = static_columns(groups, features)
    return {"static_cols": static, "temp": "TEMP" in features, "flags": ["is_virtual", "is_context"],
            "n_static": len(static), "n_temp": N_TEMP if "TEMP" in features else 0}


def _window_filter(lookback: int) -> pl.Expr:
    if lookback == 365:
        return pl.col("in_lb365")
    if lookback == 1095:
        return pl.col("in_lb1095")
    return pl.lit(True)


def build_examples(rows_t: pl.DataFrame, row_codes: pl.DataFrame, target: str, node_index: pl.DataFrame,
                   X: np.ndarray, groups: dict, g_layer: nx.Graph | nx.DiGraph, opts: Options, cfg: dict) -> list[Example]:
    cat2idx = dict(zip(node_index["category"].to_list(), node_index["idx"].to_list()))
    fc = feature_columns(groups, opts.features)
    Xs = X[:, fc["static_cols"]].astype(np.float32) if fc["static_cols"] else np.zeros((X.shape[0], 0), np.float32)
    use_temp = fc["temp"]
    recent = int(cfg["windows"]["recent_days"])
    cov_names = covariate_names(opts.features)
    drop = set(opts.drop_codes)

    keep = (pl.col("in_main_graph") & ~pl.col(f"masked_{target}") & _window_filter(opts.lookback))
    rc = row_codes.filter(keep).select("person_id", "index_date", "category", "days_since_last", "days_since_first",
                                       "in_lb365", "in_lb1095")
    dropped_counts = {}
    if drop:
        d = rc.filter(pl.col("category").is_in(list(drop)))
        dc = d.group_by("person_id", "index_date").agg(pl.col("in_lb365").sum().alias("d365"),
                                                       pl.col("in_lb1095").sum().alias("d1095"), pl.len().alias("dall"))
        dropped_counts = {(r["person_id"], r["index_date"]): r for r in dc.iter_rows(named=True)}
        rc = rc.filter(~pl.col("category").is_in(list(drop)))
    per_row = {(k[0], k[1]): grp for k, grp in rc.group_by("person_id", "index_date")}

    def nbrs(u: str) -> set[str]:
        if opts.directed:
            return set(g_layer.successors(u)) | set(g_layer.predecessors(u))
        return set(g_layer.neighbors(u))

    def edges_for(node_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        pos = {c: i for i, c in enumerate(node_list)}
        src, dst, w = [], [], []
        seen = set()
        for u in node_list:
            if u not in g_layer:
                continue
            outs = g_layer.successors(u) if opts.directed else g_layer.neighbors(u)
            for v in outs:
                if v not in pos or u == v:
                    continue
                wt = float(g_layer[u][v].get("weight", 1.0)) if opts.weighted else 1.0
                pairs = [(u, v)] if opts.directed else [(u, v), (v, u)]
                if opts.directed and opts.symmetrise:
                    pairs.append((v, u))
                for a, b in pairs:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    src.append(pos[a]); dst.append(pos[b]); w.append(wt)
        ei = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
        return ei, np.array(w, dtype=np.float32)

    out: list[Example] = []
    for r in rows_t.iter_rows(named=True):
        key = (r["person_id"], r["index_date"])
        grp = per_row.get(key)
        codes = sorted(grp["category"].to_list(), key=lambda c: cat2idx[c]) if grp is not None else []
        temp_by_code = {}
        if grp is not None and use_temp:
            for c, dl, df in zip(grp["category"], grp["days_since_last"], grp["days_since_first"]):
                temp_by_code[c] = (np.log1p(dl), np.log1p(df), 1.0 if dl <= recent else 0.0)
        context: list[str] = []
        if opts.expand and codes:
            cs = set(codes)
            ctx = set()
            for u in codes:
                if u in g_layer:
                    ctx |= nbrs(u)
            context = sorted((c for c in ctx if c not in cs and c in cat2idx), key=lambda c: cat2idx[c])
        nodes = codes + context
        if nodes:
            ids = np.array([cat2idx[c] for c in nodes], dtype=np.int64)
            parts = [Xs[ids]]
            if use_temp:
                tmp = np.zeros((len(nodes), N_TEMP), np.float32)
                for i, c in enumerate(codes):
                    tmp[i] = temp_by_code[c]
                parts.append(tmp)
            flags = np.zeros((len(nodes), N_FLAGS), np.float32)
            flags[len(codes):, 1] = 1.0
            parts.append(flags)
            x = np.concatenate(parts, axis=1).astype(np.float32)
            ei, ew = edges_for(nodes)
        else:
            ids = np.array([-1], dtype=np.int64)
            x = np.zeros((1, Xs.shape[1] + (N_TEMP if use_temp else 0) + N_FLAGS), np.float32)
            x[0, -2] = 1.0
            ei, ew = np.zeros((2, 0), dtype=np.int64), np.zeros(0, np.float32)
        cov_vals = []
        if "DEMO" in opts.features:
            cov_vals += [float(r["age_at_index"]), 1.0 if r["sex"] == "female" else 0.0]
        if "UTIL" in opts.features:
            dcnt = dropped_counts.get(key, {"d365": 0, "d1095": 0, "dall": 0})
            cov_vals += [float(r["n_codes_lb365"] - dcnt["d365"]), float(r["n_codes_lb1095"] - dcnt["d1095"]),
                         float(r["n_codes_lball"] - dcnt["dall"]), float(r["n_rows_lb1095"]), float(r["n_visit_days_365"]),
                         float(r["days_since_first_dx"]), float(r["index_year"])]
        assert len(cov_vals) == len(cov_names)
        out.append(Example(x=x, edge_index=ei, edge_weight=ew, node_ids=ids, cov=np.array(cov_vals, np.float32),
                           y=int(bool(r["positive"])), person_id=int(r["person_id"]), index_date=r["index_date"], split=r["split"]))
    return out


def fit_cov_scaler(examples: list[Example], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not names:
        return np.zeros(0, np.float32), np.ones(0, np.float32)
    M = np.stack([e.cov for e in examples]).astype(np.float64)
    mean, sd = M.mean(axis=0), M.std(axis=0)
    sd[sd == 0] = 1.0
    for j, n in enumerate(names):
        if n in UNSCALED:
            mean[j], sd[j] = 0.0, 1.0
    return mean.astype(np.float32), sd.astype(np.float32)


def apply_cov_scaler(examples: list[Example], mean: np.ndarray, sd: np.ndarray, names: list[str]) -> list[Example]:
    if not names:
        return examples
    return [replace(e, cov=((e.cov - mean) / sd).astype(np.float32)) for e in examples]
