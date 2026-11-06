"""Project 1 inputs: presence check, task-label filter, class balance."""
from __future__ import annotations

from pathlib import Path

import polars as pl

P1_FILES = [
    "p2_labels.parquet", "p2_label_counts.csv", "patient_split.csv", "target_definitions.csv", "nodes.csv",
    "graph_nodes.csv", "bridges.csv", "p2/patient_index_subgraphs.csv", "subgroup_labels.csv", "patient_category.parquet", "graph_main.graphml",
    "p2/node_features.csv", "p2/graph_backbone_topk.graphml",
]


def check(p1: Path) -> list[str]:
    """Return the Project 1 output files that are missing under p1."""
    return [f for f in P1_FILES if not (Path(p1) / f).exists()]


def task_labels(labels: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Filter p2_labels to the task: one definition, one horizon, optionally uncensored rows only."""
    t = cfg["task"]
    out = labels.filter((pl.col("definition") == t["definition"]) & (pl.col("horizon_months") == int(t["horizon_months"])))
    if t.get("exclude_censored", True):
        out = out.filter(~pl.col("censored"))
    return out.drop("definition", "horizon_months", "censored")


def class_balance(labels: pl.DataFrame) -> pl.DataFrame:
    agg = [pl.len().alias("rows"), pl.col("person_id").n_unique().alias("patients"), pl.col("positive").sum().alias("positives")]
    per = labels.group_by("target", "split").agg(*agg)
    tot = labels.group_by("target").agg(*agg).with_columns(pl.lit("all").alias("split")).select(per.columns)
    return (pl.concat([per, tot]).with_columns((pl.col("positives") / pl.col("rows")).alias("positive_rate"))
            .sort("target", "split"))
