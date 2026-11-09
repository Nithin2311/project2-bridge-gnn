"""Step 04: 3-node directed motif census on Layer D and bridge-centred diagnosis chains.

Feed-forward triads (NetworkX class 030T: a -> b, b -> c, a -> c) are the "sequence" structure Dr. Kim asked
about. Two nulls are reported: a free directed configuration model (degrees kept, any pair allowed) and a
Layer-U-restricted one (degrees kept, new edges only on validated undirected pairs). Layer D is a DAG by
construction (one direction per pair, oriented by majority first-record order), so every skeleton triangle is
030T; the restricted null is the one that separates direction from Layer U's clustering. Chains are counted
directly on patients' first-record dates and are statements about recorded order, not about the graph.
"""
from __future__ import annotations

import random

import networkx as nx
import numpy as np
import polars as pl

TRIAD_CLASSES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U",
                 "120C", "210", "300"]
CONNECTED = {"021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"}


def triad_census(dg: nx.DiGraph) -> dict[str, int]:
    return {k: int(v) for k, v in nx.triadic_census(dg).items()}


def _rewire(dg: nx.DiGraph, seed: int, swaps_per_edge: int = 10, no_reciprocal: bool | None = None,
            allowed_pairs: set[frozenset] | None = None) -> nx.DiGraph:
    """Directed degree-preserving rewiring by 2-edge swaps: (a->b, c->d) -> (a->d, c->b) when that creates no
    self-loop or duplicate edge. Keeps every node's in- and out-degree. (NetworkX's directed_edge_swap needs
    3-edge paths and stalls on sparse graphs.) When the input has no reciprocal pairs (Layer D by construction),
    swaps that would create one are refused so the null stays in the same space. With `allowed_pairs`, new
    edges must lie on those undirected pairs (the Layer-U-restricted null)."""
    rng = random.Random(seed)
    h = dg.copy()
    if no_reciprocal is None:
        no_reciprocal = not any(h.has_edge(v, u) for u, v in h.edges())
    edges = list(h.edges())
    m = len(edges)
    if m < 2:
        return h
    target, tries, done = swaps_per_edge * m, 0, 0
    while done < target and tries < 100 * m:
        tries += 1
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b or h.has_edge(a, d) or h.has_edge(c, b):
            continue
        if no_reciprocal and (h.has_edge(d, a) or h.has_edge(b, c)):
            continue
        if allowed_pairs is not None and (frozenset((a, d)) not in allowed_pairs or frozenset((c, b)) not in allowed_pairs):
            continue
        h.remove_edge(a, b)
        h.remove_edge(c, d)
        h.add_edge(a, d)
        h.add_edge(c, b)
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    return h


def _z_p(obs: np.ndarray, nulls: np.ndarray) -> tuple[list, list]:
    """z for connected classes with non-zero null SD; two-sided empirical p with the +1 correction."""
    n = nulls.shape[0]
    mean, sd = nulls.mean(axis=0), (nulls.std(axis=0, ddof=1) if n > 1 else np.zeros(nulls.shape[1]))
    z, p = [], []
    for k, o, mu, s, col in zip(TRIAD_CLASSES, obs, mean, sd, nulls.T):
        if k in CONNECTED and s > 0:
            z.append(float((o - mu) / s))
            hi = (np.sum(col >= o) + 1) / (n + 1)
            lo = (np.sum(col <= o) + 1) / (n + 1)
            p.append(float(min(1.0, 2 * min(hi, lo))))
        else:
            z.append(None)
            p.append(None)
    return z, p


def null_census(dg: nx.DiGraph, n: int, seed: int, allowed_pairs: set[frozenset] | None = None) -> pl.DataFrame:
    """Observed census vs n free rewirings (columns null_*, z, p_emp) and, if `allowed_pairs` is given, vs n
    Layer-U-restricted rewirings (null_u_*, z_u, p_emp_u)."""
    obs_d = triad_census(dg)
    obs = np.array([obs_d[k] for k in TRIAD_CLASSES], dtype=float)
    free = np.array([[triad_census(_rewire(dg, seed + i))[k] for k in TRIAD_CLASSES] for i in range(n)], dtype=float)
    z, p = _z_p(obs, free)
    out = pl.DataFrame({"triad": TRIAD_CLASSES, "observed": obs.astype(int).tolist(),
                        "null_mean": free.mean(axis=0).tolist(),
                        "null_sd": (free.std(axis=0, ddof=1) if n > 1 else np.zeros(len(TRIAD_CLASSES))).tolist(),
                        "z": pl.Series(z, dtype=pl.Float64), "p_emp": pl.Series(p, dtype=pl.Float64)})
    if allowed_pairs is not None:
        restr = np.array([[triad_census(_rewire(dg, seed + 10_000 + i, allowed_pairs=allowed_pairs))[k]
                           for k in TRIAD_CLASSES] for i in range(n)], dtype=float)
        z_u, p_u = _z_p(obs, restr)
        out = out.with_columns(pl.Series("null_u_mean", restr.mean(axis=0).tolist()),
                               pl.Series("null_u_sd", (restr.std(axis=0, ddof=1) if n > 1 else np.zeros(len(TRIAD_CLASSES))).tolist()),
                               pl.Series("z_u", z_u, dtype=pl.Float64), pl.Series("p_emp_u", p_u, dtype=pl.Float64))
    return out


def feed_forward_middle_share(dg: nx.DiGraph, codes: set[str], cores: dict[str, set[str]] | None = None) -> pl.DataFrame:
    """Per code: feed-forward triads (a -> code -> c with a -> c, no reverse edges) where it is the middle; with
    `cores`, also those whose end node c is in a target core and whose start a is not."""
    total = triad_census(dg)["030T"]
    cores = cores or {}
    rows = []
    for b in sorted(codes):
        n_mid = 0
        into = {t: 0 for t in cores}
        if b in dg:
            for a in dg.predecessors(b):
                if dg.has_edge(b, a):
                    continue
                for c in dg.successors(b):
                    if a == c or dg.has_edge(c, b) or not dg.has_edge(a, c) or dg.has_edge(c, a):
                        continue
                    n_mid += 1
                    for t, members in cores.items():
                        if c in members and a not in members:
                            into[t] += 1
        rows.append({"code": b, "n_middle": n_mid, "share_of_ff": (n_mid / total) if total else 0.0,
                     **{f"n_middle_into_core_{t}": v for t, v in into.items()}})
    return pl.DataFrame(rows)


def bridge_chains(fd: pl.DataFrame, bridges: set[str], cores: dict[str, set[str]], gap_days: int,
                  min_patients: int, fd_full: pl.DataFrame | None = None, node_set: set[str] | None = None) -> pl.DataFrame:
    """Recorded-order triples a -> bridge -> core code per target.

    n_patients: patients with first(a) + gap <= first(bridge) <= first(core) - gap (strict '<' when gap is 0).
    n_patients_entry: the subset with NO code of that target's core dated before first(bridge), judged on
    `fd_full` (first dates before any day-one drop) so that problem-list carry-over counts as prior membership.
    a_in_graph: whether `a` is a main-graph node (R/Z/O symptom and status codes are not, yet can start chains).
    """
    full = fd_full if fd_full is not None else fd
    fb = fd.filter(pl.col("category").is_in(list(bridges))).rename({"category": "bridge", "first_date": "d_bridge"})

    def ordered(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
        diff = (later - earlier).dt.total_days()
        return diff >= gap_days if gap_days > 0 else diff > 0

    parts = []
    for t, members in cores.items():
        fc = fd.filter(pl.col("category").is_in(list(members))).rename({"category": "core_code", "first_date": "d_core"})
        fa = fd.filter(~pl.col("category").is_in(list(members | bridges))).rename({"category": "a", "first_date": "d_a"})
        prior = (full.filter(pl.col("category").is_in(list(members)))
                 .group_by("person_id").agg(pl.col("first_date").min().alias("d_core_any")))
        j = (fa.join(fb, on="person_id").join(fc, on="person_id")
             .filter(ordered(pl.col("d_bridge"), pl.col("d_a")) & ordered(pl.col("d_core"), pl.col("d_bridge")))
             .join(prior, on="person_id", how="left")
             .with_columns((pl.col("d_core_any").is_null() | (pl.col("d_core_any") >= pl.col("d_bridge"))).alias("entry")))
        parts.append(j.group_by("a", "bridge", "core_code").agg(
            pl.col("person_id").n_unique().alias("n_patients"),
            pl.col("person_id").filter(pl.col("entry")).n_unique().alias("n_patients_entry"))
            .with_columns(pl.lit(t).alias("target")))
    if not parts:
        return pl.DataFrame(schema={"target": pl.Utf8, "a": pl.Utf8, "a_in_graph": pl.Boolean, "bridge": pl.Utf8,
                                    "core_code": pl.Utf8, "n_patients": pl.UInt32, "n_patients_entry": pl.UInt32})
    out = pl.concat(parts)
    nodes = node_set if node_set is not None else set(fd["category"].to_list())
    out = out.with_columns(pl.col("a").is_in(list(nodes)).alias("a_in_graph"))
    return (out.filter(pl.col("n_patients") >= min_patients)
            .select("target", "a", "a_in_graph", "bridge", "core_code", "n_patients", "n_patients_entry")
            .sort("n_patients", "target", "a", descending=[True, False, False]))
