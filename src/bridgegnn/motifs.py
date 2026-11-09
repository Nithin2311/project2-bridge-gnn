"""Step 04: 3-node directed motif census on Layer D and bridge-centred diagnosis chains.

Feed-forward triads (NetworkX class 030T: a -> b, b -> c, a -> c) are the "sequence" structure Dr. Kim asked
about; the null is directed degree-preserving rewiring. Chains are counted directly on patients' first-record
dates, so they are a statement about recorded order, not about the graph.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl

TRIAD_CLASSES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U",
                 "120C", "210", "300"]
CONNECTED = {"021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"}


def triad_census(dg: nx.DiGraph) -> dict[str, int]:
    return {k: int(v) for k, v in nx.triadic_census(dg).items()}


def _rewire(dg: nx.DiGraph, seed: int, swaps_per_edge: int = 10, no_reciprocal: bool | None = None) -> nx.DiGraph:
    """Directed degree-preserving rewiring by 2-edge swaps: (a->b, c->d) -> (a->d, c->b) when that creates no
    self-loop or duplicate edge. Keeps every node's in- and out-degree. (NetworkX's directed_edge_swap needs
    3-edge paths and stalls on sparse graphs.) When the input has no reciprocal pairs (Layer D by construction),
    swaps that would create one are refused so the null stays in the same space."""
    import random

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
            continue  # Layer D holds one direction per pair; keep the null in the same space
        h.remove_edge(a, b)
        h.remove_edge(c, d)
        h.add_edge(a, d)
        h.add_edge(c, b)
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    return h


def null_census(dg: nx.DiGraph, n: int, seed: int) -> pl.DataFrame:
    obs = triad_census(dg)
    nulls = np.array([[triad_census(_rewire(dg, seed + i))[k] for k in TRIAD_CLASSES] for i in range(n)], dtype=float)
    mean, sd = nulls.mean(axis=0), nulls.std(axis=0, ddof=1) if n > 1 else np.zeros(len(TRIAD_CLASSES))
    z = [float((obs[k] - mu) / s) if s > 0 else None for k, mu, s in zip(TRIAD_CLASSES, mean, sd)]
    return pl.DataFrame({"triad": TRIAD_CLASSES, "observed": [obs[k] for k in TRIAD_CLASSES],
                         "null_mean": mean.tolist(), "null_sd": sd.tolist(), "z": pl.Series(z, dtype=pl.Float64)})


def feed_forward_middle_share(dg: nx.DiGraph, codes: set[str]) -> pl.DataFrame:
    """For each code: number of feed-forward triads (a -> code -> c with a -> c, no reverse edges) where it is the middle."""
    total = triad_census(dg)["030T"]
    rows = []
    for b in sorted(codes):
        n_mid = 0
        if b in dg:
            for a in dg.predecessors(b):
                if dg.has_edge(b, a):
                    continue
                for c in dg.successors(b):
                    if a == c or dg.has_edge(c, b):
                        continue
                    if dg.has_edge(a, c) and not dg.has_edge(c, a):
                        n_mid += 1
        rows.append({"code": b, "n_middle": n_mid, "share_of_ff": (n_mid / total) if total else 0.0})
    return pl.DataFrame(rows)


def bridge_chains(fd: pl.DataFrame, bridges: set[str], cores: dict[str, set[str]], gap_days: int,
                  min_patients: int) -> pl.DataFrame:
    """Patients with first(a) + gap <= first(bridge) and first(bridge) + gap <= first(core code), per target."""
    fb = fd.filter(pl.col("category").is_in(list(bridges))).rename({"category": "bridge", "first_date": "d_bridge"})
    parts = []
    for t, members in cores.items():
        fc = (fd.filter(pl.col("category").is_in(list(members)))
              .rename({"category": "core_code", "first_date": "d_core"}))
        fa = (fd.filter(~pl.col("category").is_in(list(members | bridges)))
              .rename({"category": "a", "first_date": "d_a"}))
        j = (fa.join(fb, on="person_id").join(fc, on="person_id")
             .filter(((pl.col("d_bridge") - pl.col("d_a")).dt.total_days() >= gap_days)
                     & ((pl.col("d_core") - pl.col("d_bridge")).dt.total_days() >= gap_days)))
        parts.append(j.group_by("a", "bridge", "core_code").agg(pl.col("person_id").n_unique().alias("n_patients"))
                     .with_columns(pl.lit(t).alias("target")))
    out = pl.concat(parts) if parts else pl.DataFrame(schema={"a": pl.Utf8, "bridge": pl.Utf8, "core_code": pl.Utf8, "n_patients": pl.UInt32, "target": pl.Utf8})
    return (out.filter(pl.col("n_patients") >= min_patients).select("target", "a", "bridge", "core_code", "n_patients")
            .sort("n_patients", "target", "a", descending=[True, False, False]))
