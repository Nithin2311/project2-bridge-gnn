"""Layer D: directed first-record precedence edges on top of the validated Project 1 edges.

For an undirected edge (a, b), count development patients whose first a precedes their first b by at least
`gap_days` (n_ab), the reverse (n_ba), and ties. A two-sided binomial test of n_ab vs n_ba at p = 0.5 with
BH-FDR over all tested pairs decides whether the pair carries a direction. Day-one codes (a patient's codes on
their first ICD-10-era day) are dropped before counting in the primary run (left truncation at the ICD-10 switch).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from .contract import SCHEMAS


def first_dates(pc: pl.DataFrame, split: pl.DataFrame, dev_splits: list[str], drop_day_one: bool) -> pl.DataFrame:
    """(person_id, category, first_date) for development patients; optionally without day-one codes."""
    dev_ids = split.filter(pl.col("split").is_in(dev_splits))["person_id"].to_list()
    fd = pc.select("person_id", "category", "first_date").filter(pl.col("person_id").is_in(dev_ids))
    test_ids = set(split.filter(pl.col("split") == "test")["person_id"].to_list())
    assert not (set(fd["person_id"].to_list()) & test_ids), "test patients in first_dates"
    if drop_day_one:
        d0 = fd.group_by("person_id").agg(pl.col("first_date").min().alias("_d0"))
        fd = fd.join(d0, on="person_id").filter(pl.col("first_date") > pl.col("_d0")).drop("_d0")
    return fd.sort("person_id", "category")


def precedence_counts(fd: pl.DataFrame, pairs: list[tuple[str, str]], gap_days: int) -> pl.DataFrame:
    """Per pair (a, b) in the given order: n_ab (a precedes b by >= gap days), n_ba, n_tie (both codes, |diff| < gap).
    With gap 0: strict precedence either way, same-day = tie."""
    pair_df = pl.DataFrame({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
    fa = fd.rename({"category": "a", "first_date": "da"})
    fb = fd.rename({"category": "b", "first_date": "db"})
    both = pair_df.join(fa, on="a", how="inner").join(fb, on=["b", "person_id"], how="inner")
    diff = (pl.col("db") - pl.col("da")).dt.total_days()
    agg = both.group_by("a", "b").agg(
        (diff >= gap_days).sum().cast(pl.Int32).alias("n_ab"),
        (diff <= -gap_days).sum().cast(pl.Int32).alias("n_ba"),
        ((diff < gap_days) & (diff > -gap_days)).sum().cast(pl.Int32).alias("n_tie"),
    ) if gap_days > 0 else both.group_by("a", "b").agg(
        (diff > 0).sum().cast(pl.Int32).alias("n_ab"),
        (diff < 0).sum().cast(pl.Int32).alias("n_ba"),
        (diff == 0).sum().cast(pl.Int32).alias("n_tie"),
    )
    out = pair_df.join(agg, on=["a", "b"], how="left").fill_null(0)
    return out.with_columns(pl.col("n_ab").cast(pl.Int32), pl.col("n_ba").cast(pl.Int32), pl.col("n_tie").cast(pl.Int32))


def directed_edges(counts: pl.DataFrame, min_total: int, alpha: float) -> pl.DataFrame:
    """Binomial test per pair with enough support, BH over tested pairs, majority direction as src -> dst."""
    tested = counts.filter((pl.col("n_ab") + pl.col("n_ba")) >= min_total)
    empty = pl.DataFrame({c: pl.Series([], dtype=t) for c, t in SCHEMAS["edges_directed"].items() if c != "phi"})
    if tested.height == 0:
        return empty
    p = np.array([binomtest(int(k), int(k + m), 0.5, alternative="two-sided").pvalue
                  for k, m in zip(tested["n_ab"], tested["n_ba"])])
    q = multipletests(p, method="fdr_bh")[1]
    tested = tested.with_columns(pl.Series("p_binom", p), pl.Series("q_bh", q))
    fwd = pl.col("n_ab") >= pl.col("n_ba")
    out = tested.with_columns(
        pl.when(fwd).then(pl.col("a")).otherwise(pl.col("b")).alias("src"),
        pl.when(fwd).then(pl.col("b")).otherwise(pl.col("a")).alias("dst"),
        pl.when(fwd).then(pl.col("n_ab")).otherwise(pl.col("n_ba")).alias("n_fwd"),
        pl.when(fwd).then(pl.col("n_ba")).otherwise(pl.col("n_ab")).alias("n_rev"),
    ).with_columns((pl.col("n_fwd") / (pl.col("n_fwd") + pl.col("n_rev"))).alias("share"))
    out = (out.filter(pl.col("q_bh") < alpha)
           .with_columns(pl.col("n_fwd").cast(pl.Int32), pl.col("n_rev").cast(pl.Int32), pl.col("n_tie").cast(pl.Int32),
                         pl.col("share").cast(pl.Float64))
           .select("src", "dst", "n_fwd", "n_rev", "n_tie", "share", "p_binom", "q_bh"))
    return out.sort("q_bh", "src", "dst") if out.height else empty


def attach_phi(dedges: pl.DataFrame, g_u: nx.Graph) -> pl.DataFrame:
    phi = [float(g_u[u][v]["weight"]) if g_u.has_edge(u, v) else float("nan") for u, v in zip(dedges["src"], dedges["dst"])]
    return dedges.with_columns(pl.Series("phi", phi, dtype=pl.Float64))


def to_digraph(dedges: pl.DataFrame, g_u: nx.Graph) -> nx.DiGraph:
    dg = nx.DiGraph()
    dg.add_nodes_from(g_u.nodes(data=True))
    for r in dedges.iter_rows(named=True):
        dg.add_edge(r["src"], r["dst"], weight=float(r["share"]), phi=float(r["phi"]), distance=1.0 / float(r["phi"]),
                    n_fwd=int(r["n_fwd"]), n_rev=int(r["n_rev"]), q_bh=float(r["q_bh"]))
    dg.graph.update(g_u.graph)
    return dg


def summary(dedges: pl.DataFrame, g_u: nx.Graph, n_tested: int) -> dict:
    """n_tested = pairs that reached the support floor; share_undetermined = tested pairs with no significant direction."""
    m = g_u.number_of_edges()
    return {"n_directed": dedges.height, "n_undirected": m, "share_directed": dedges.height / m if m else 0.0,
            "median_share": float(dedges["share"].median()) if dedges.height else float("nan"),
            "n_tested": n_tested, "share_undetermined": (1 - dedges.height / n_tested) if n_tested else float("nan")}


def edges_into_cores(dedges: pl.DataFrame, cores: dict[str, set[str]], top_k: int) -> pl.DataFrame:
    parts = []
    for t, members in cores.items():
        sub = dedges.filter(pl.col("dst").is_in(list(members)) & ~pl.col("src").is_in(list(members)))
        parts.append(sub.sort("n_fwd", descending=True).head(top_k).with_columns(pl.lit(t).alias("target")))
    cols = ["target", "src", "dst", "n_fwd", "n_rev", "share", "q_bh"]
    return pl.concat(parts).select(cols) if parts else pl.DataFrame(schema={c: pl.Utf8 for c in cols})


def edges_through_bridges(dedges: pl.DataFrame, bridge_codes: set[str]) -> pl.DataFrame:
    b = list(bridge_codes)
    into = dedges.filter(pl.col("dst").is_in(b)).with_columns(pl.lit("into_bridge").alias("role"), pl.col("dst").alias("bridge"))
    out = dedges.filter(pl.col("src").is_in(b)).with_columns(pl.lit("out_of_bridge").alias("role"), pl.col("src").alias("bridge"))
    return pl.concat([into, out]).sort("bridge", "role", pl.col("n_fwd"), descending=[False, False, True])
