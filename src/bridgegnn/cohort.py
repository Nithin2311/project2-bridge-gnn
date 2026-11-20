"""Aggregate cohort tables for the report: rows / patients / positives per target and split, test-set composition,
the distribution of the triggering (first in-horizon core) code among positives, and masked subgraph sizes.
Everything returned is a count or a share; no patient-level value leaves these functions."""
from __future__ import annotations

import polars as pl


def cohort_table(rows: pl.DataFrame, dev_splits: tuple[str, ...] = ("train", "val")) -> pl.DataFrame:
    """One row per target: development and test rows, patients and positives, and the test set's female and 65+ shares."""
    unknown = set(rows["split"].unique().to_list()) - set(dev_splits) - {"test"}
    if unknown:
        raise ValueError(f"unexpected split values {sorted(unknown)}")
    r = rows.with_columns(pl.when(pl.col("split").is_in(list(dev_splits))).then(pl.lit("dev")).otherwise(pl.lit("test")).alias("part"))
    parts = r.group_by("target", "part").agg(pl.len().alias("n_rows"), pl.col("person_id").n_unique().alias("n_patients"),
                                            pl.col("positive").sum().alias("n_pos"))
    wide = parts.pivot(on="part", index="target", values=["n_rows", "n_patients", "n_pos"])
    test = r.filter(pl.col("part") == "test").group_by("target").agg(
        (pl.col("sex") == "female").mean().alias("test_female_share"), (pl.col("age_at_index") >= 65).mean().alias("test_age65_share"),
        pl.col("age_at_index").median().alias("test_age_median"))
    out = wide.join(test, on="target", how="left").sort("target")
    for c in ["n_rows_dev", "n_patients_dev", "n_pos_dev", "n_rows_test", "n_patients_test", "n_pos_test"]:
        if c not in out.columns:
            out = out.with_columns(pl.lit(0).alias(c))
    return out.with_columns([pl.col(c).cast(pl.Int64) for c in ["n_rows_dev", "n_patients_dev", "n_pos_dev", "n_rows_test", "n_patients_test", "n_pos_test"]])


def triggering_codes(pos_rows: pl.DataFrame, first_dates: pl.DataFrame, core: dict[str, list[str]], horizon_days: int) -> pl.DataFrame:
    """For every positive row, the earliest core code first recorded inside (index, index + horizon]; counts and shares per target."""
    out = []
    for target, members in core.items():
        sub = pos_rows.filter(pl.col("target") == target).select("person_id", "index_date")
        if sub.height == 0:
            continue
        j = sub.join(first_dates.filter(pl.col("category").is_in(members)), on="person_id", how="inner")
        j = j.with_columns((pl.col("first_date") - pl.col("index_date")).dt.total_days().alias("days"))
        j = j.filter((pl.col("days") > 0) & (pl.col("days") <= horizon_days))
        # same-day ties share the credit 1/k, so the attribution does not depend on row order
        ranked = j.with_columns(pl.col("days").min().over("person_id", "index_date").alias("first_day")).filter(pl.col("days") == pl.col("first_day"))
        ranked = ranked.with_columns((1.0 / pl.len().over("person_id", "index_date")).alias("credit"))
        n_tied = int((ranked.group_by("person_id", "index_date").len().filter(pl.col("len") > 1)).height)
        counts = ranked.group_by("category").agg(pl.col("credit").sum().alias("n")).sort("n", "category", descending=[True, False])
        total = float(counts["n"].sum())
        out.append(counts.with_columns(pl.lit(target).alias("target"), (pl.col("n") / total).alias("share"),
                                       pl.lit(sub.height).alias("n_positive_rows"), pl.lit(round(total)).alias("n_resolved"), pl.lit(n_tied).alias("n_tied")))
    return pl.concat(out).select("target", "category", "n", "share", "n_positive_rows", "n_resolved", "n_tied") if out else pl.DataFrame()


def subgraph_sizes(rows: pl.DataFrame, row_codes: pl.DataFrame, targets: list[str], window_col: str = "in_lb1095") -> pl.DataFrame:
    """Per target: median, quartiles and share-empty of the number of unmasked main-graph codes in the window (the GNN's nodes)."""
    out = []
    for t in targets:
        keep = row_codes.filter(pl.col(window_col) & pl.col("in_main_graph") & ~pl.col(f"masked_{t}"))
        n = keep.group_by("person_id", "index_date").len().rename({"len": "n_nodes"})
        r = rows.filter(pl.col("target") == t).select("person_id", "index_date").join(n, on=["person_id", "index_date"], how="left").fill_null(0)
        s = r["n_nodes"]
        out.append({"target": t, "n_rows": r.height, "median_nodes": float(s.median()), "q1_nodes": float(s.quantile(0.25)),
                    "q3_nodes": float(s.quantile(0.75)), "mean_nodes": float(s.mean()), "share_empty": float((s == 0).mean())})
    return pl.DataFrame(out)


def core_members(target_definitions: pl.DataFrame, node_features: pl.DataFrame) -> pl.DataFrame:
    """Core members per target with their development prevalence and chapter-block status (for the supplement)."""
    prev = node_features.select("category", "prevalence") if "prevalence" in node_features.columns else node_features.select("category")
    core = target_definitions.filter(pl.col("in_core")).select("target", "category", "name", "in_chapter_block")
    return core.join(prev, on="category", how="left").sort("target", "prevalence", descending=[False, True])


def bridge_carriers(rows: pl.DataFrame, row_codes: pl.DataFrame, bridges_by_target: dict[str, list[str]],
                    window_col: str = "in_lb1095") -> pl.DataFrame:
    """Per target and survivor bridge code: eligible rows (and positive rows) that carry the code inside the window."""
    out = []
    for t, codes in bridges_by_target.items():
        r = rows.filter(pl.col("target") == t).select("person_id", "index_date", "positive")
        carried = row_codes.filter(pl.col(window_col) & pl.col("category").is_in(codes) & ~pl.col(f"masked_{t}")).select("person_id", "index_date", "category").unique()
        j = r.join(carried, on=["person_id", "index_date"], how="inner")
        counts = j.group_by("category").agg(pl.len().alias("n_rows"), pl.col("positive").sum().alias("n_pos_rows"))
        for c in codes:
            m = counts.filter(pl.col("category") == c)
            out.append({"target": t, "category": c, "n_rows": int(m["n_rows"][0]) if m.height else 0,
                        "n_pos_rows": int(m["n_pos_rows"][0]) if m.height else 0, "n_rows_total": r.height, "n_pos_total": int(r["positive"].sum())})
    return pl.DataFrame(out)
