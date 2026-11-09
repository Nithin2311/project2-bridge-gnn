"""Static node feature matrix from Project 1 artifacts (Nithin's cross-check copy of step 02's node part).

Columns, in order: STRUCT (P1 structural measures), BRIDGE (survivor flag, per-target flag and score),
CHAPTER (one-hot of the ICD chapter letter), COMMUNITY (one-hot of the seed-0 Leiden community, small ones
grouped as `community_other`). Standardised over the 496 nodes (train+val artifact; no split inside).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .contract import SCHEMAS

STRUCT_COLS = ["prevalence", "degree", "weighted_degree", "betweenness", "bt_null_percentile", "closeness",
               "eigenvector", "pagerank", "participation", "within_module_z", "degree_residual_z",
               "inpatient_share", "ccir_chronic_share", "mean_age_first_occurrence", "female_share", "backbone_degree"]


def _pairs_with(target: str, pairs: pl.Series) -> pl.Expr:
    return pl.col("pair").str.split("-").list.contains(target)


def bridge_table(node_features: pl.DataFrame, bridges: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    """category, bridge_flag, bridge_flag_<t>, bridge_score_<t> for every node."""
    out = node_features.select("category", pl.col("bridge_survivor").cast(pl.Float64).alias("bridge_flag"))
    for t in targets:
        sub = bridges.filter(_pairs_with(t, bridges["pair"]))
        agg = sub.group_by("category").agg(pl.col("bridge_pass").any().alias(f"bridge_flag_{t}"),
                                          pl.col("bridge_betweenness").max().alias(f"bridge_score_{t}"))
        out = out.join(agg, on="category", how="left").with_columns(
            pl.col(f"bridge_flag_{t}").fill_null(False).cast(pl.Float64),
            pl.col(f"bridge_score_{t}").fill_null(0.0).cast(pl.Float64))
    return out


def bflag_codes(node_features: pl.DataFrame, bridges: pl.DataFrame, target: str) -> set[str]:
    bt = bridge_table(node_features, bridges, [target])
    return set(bt.filter(pl.col(f"bridge_flag_{target}") > 0)["category"].to_list())


def build(node_features: pl.DataFrame, bridges: pl.DataFrame, targets: list[str], min_community_size: int = 5,
          standardise: bool = True) -> tuple[pl.DataFrame, np.ndarray, dict[str, list[str]]]:
    nf = node_features.sort("category")
    cats = nf["category"].to_list()
    groups: dict[str, list[str]] = {}
    blocks: list[np.ndarray] = []

    struct = nf.select([pl.col(c).cast(pl.Float64) for c in STRUCT_COLS]).to_numpy()
    groups["STRUCT"] = list(STRUCT_COLS)
    blocks.append(struct)

    bt = bridge_table(nf, bridges, targets).sort("category")
    assert bt["category"].to_list() == cats
    bcols = [c for c in bt.columns if c != "category"]
    groups["BRIDGE"] = bcols
    blocks.append(bt.select(bcols).to_numpy().astype(np.float64))

    letters = [c[0] for c in cats]
    uniq_letters = sorted(set(letters))
    groups["CHAPTER"] = [f"chapter_{letter}" for letter in uniq_letters]
    blocks.append(np.array([[1.0 if le == u else 0.0 for u in uniq_letters] for le in letters]))

    comm = nf["community_seed0"].to_list()
    counts = {c: comm.count(c) for c in set(comm)}
    big = sorted(c for c, n in counts.items() if n >= min_community_size)
    groups["COMMUNITY"] = [f"community_{c}" for c in big] + ["community_other"]
    blocks.append(np.array([[1.0 if cm == b else 0.0 for b in big] + [0.0 if cm in big else 1.0] for cm in comm]))

    X = np.concatenate(blocks, axis=1)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    if standardise:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    node_index = pl.DataFrame({"category": cats, "idx": pl.Series(list(range(len(cats))), dtype=pl.Int32)})
    return node_index, X.astype(np.float32), groups


def write(node_index: pl.DataFrame, X: np.ndarray, groups: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from .contract import write as cwrite
    cwrite(node_index, out_dir / "node_index.csv", "node_index")
    np.save(out_dir / "node_x.npy", X.astype(np.float32))
    (out_dir / "feature_groups.json").write_text(json.dumps(groups, indent=1))


def read(d: Path) -> tuple[pl.DataFrame, np.ndarray, dict]:
    d = Path(d)
    from .contract import read as cread
    idx = cread(d / "node_index.csv", "node_index")
    X = np.load(d / "node_x.npy")
    groups = json.loads((d / "feature_groups.json").read_text())
    assert X.shape[0] == idx.height and list(SCHEMAS["node_index"]) == idx.columns
    return idx, X, groups
