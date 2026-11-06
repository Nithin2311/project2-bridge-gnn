"""Step 01 pure functions: per-row code sets with window flags and target masks, and the row table.

Inputs are polars frames (see tests/conftest.py and synth.py for shapes); no database access here.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from .contract import SCHEMAS, TARGETS


def mask_sets(target_defs: pl.DataFrame, cfg: dict) -> dict[str, set[str]]:
    """Per target, the categories excluded from that target's inputs (core plus, by config, chapter block)."""
    excl = cfg["masking"]["input_exclusion"]
    cols = ["in_core"]
    if "chapter_block" in excl:
        cols.append("in_chapter_block")
    if "community" in excl:
        cols.append("in_community")
    out: dict[str, set[str]] = {}
    for t in target_defs["target"].unique().to_list():
        sub = target_defs.filter(pl.col("target") == t)
        out[t] = set(sub.filter(pl.any_horizontal([pl.col(c) for c in cols]))["category"].to_list())
    return out


def _days(expr_a: pl.Expr, expr_b: pl.Expr) -> pl.Expr:
    return (expr_a - expr_b).dt.total_days().cast(pl.Int32)


def build_row_codes(index_rows: pl.DataFrame, codes_long: pl.DataFrame, nodes: pl.DataFrame,
                    masks: dict[str, set[str]], cfg: dict) -> pl.DataFrame:
    """One row per (person_id, index_date, category) recorded in the era on or before the index."""
    era_start = dt.date.fromisoformat(str(cfg["windows"]["era_start"]))
    idx = index_rows.select("person_id", "index_date").unique()
    era = codes_long.filter(pl.col("condition_date") >= era_start)
    joined = idx.join(era, on="person_id", how="inner").filter(pl.col("condition_date") <= pl.col("index_date"))
    rc = joined.group_by("person_id", "index_date", "category").agg(
        pl.col("condition_date").min().alias("first_date"), pl.col("condition_date").max().alias("last_date"))
    node_set = set(nodes["category"].to_list())
    rc = rc.with_columns(
        _days(pl.col("index_date"), pl.col("first_date")).alias("days_since_first"),
        _days(pl.col("index_date"), pl.col("last_date")).alias("days_since_last"),
        (pl.col("last_date") > pl.col("index_date") - pl.duration(days=365)).alias("in_lb365"),
        (pl.col("last_date") > pl.col("index_date") - pl.duration(days=1095)).alias("in_lb1095"),
        pl.col("category").is_in(list(node_set)).alias("in_main_graph"),
        *[pl.col("category").is_in(list(masks.get(t, set()))).alias(f"masked_{t}") for t in TARGETS],
    )
    return rc.select(list(SCHEMAS["row_codes"])).sort("person_id", "index_date", "category")


def build_rows(labels: pl.DataFrame, row_codes: pl.DataFrame, visits: pl.DataFrame, demo: pl.DataFrame,
               codes_long: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """The task rows with patient covariates. Window counts are over main-graph codes BEFORE masking."""
    w = cfg["windows"]
    era_start = dt.date.fromisoformat(str(w["era_start"]))
    vis_days = int(w["visit_window_days"])
    base = labels.select("person_id", "index_date", "target", "split", "positive", "days_to_event")
    idx = base.select("person_id", "index_date").unique()

    code_stats = row_codes.group_by("person_id", "index_date").agg(
        (pl.col("in_lb365") & pl.col("in_main_graph")).sum().alias("n_codes_lb365"),
        (pl.col("in_lb1095") & pl.col("in_main_graph")).sum().alias("n_codes_lb1095"),
        pl.col("in_main_graph").sum().alias("n_codes_lball"))

    era = codes_long.filter(pl.col("condition_date") >= era_start)
    row_counts = (idx.join(era, on="person_id", how="inner")
                  .filter((pl.col("condition_date") <= pl.col("index_date"))
                          & (pl.col("condition_date") > pl.col("index_date") - pl.duration(days=1095)))
                  .group_by("person_id", "index_date").agg(pl.len().alias("n_rows_lb1095")))
    first_dx = era.group_by("person_id").agg(pl.col("condition_date").min().alias("first_dx"))
    visit_counts = (idx.join(visits, on="person_id", how="inner")
                    .filter((pl.col("visit_date") <= pl.col("index_date"))
                            & (pl.col("visit_date") > pl.col("index_date") - pl.duration(days=vis_days)))
                    .group_by("person_id", "index_date").agg(pl.col("visit_date").n_unique().alias("n_visit_days_365")))

    out = (base.join(demo.select("person_id", "sex", "birth_date"), on="person_id", how="left")
           .join(first_dx, on="person_id", how="left")
           .join(code_stats, on=["person_id", "index_date"], how="left")
           .join(row_counts, on=["person_id", "index_date"], how="left")
           .join(visit_counts, on=["person_id", "index_date"], how="left"))
    out = out.with_columns(
        ((pl.col("index_date") - pl.col("birth_date")).dt.total_days() / 365.25).alias("age_at_index"),
        pl.col("index_date").dt.year().cast(pl.Int32).alias("index_year"),
        _days(pl.col("index_date"), pl.col("first_dx")).alias("days_since_first_dx"),
        *[pl.col(c).fill_null(0).cast(pl.Int32) for c in
          ["n_codes_lb365", "n_codes_lb1095", "n_codes_lball", "n_rows_lb1095", "n_visit_days_365"]],
        pl.col("days_to_event").cast(pl.Int64),
    )
    return out.select(list(SCHEMAS["rows"])).sort("target", "person_id", "index_date")
