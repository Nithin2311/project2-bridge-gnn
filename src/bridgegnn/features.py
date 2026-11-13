"""Step 02: shared node features and target-masked tabular feature matrices."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from . import contract, nodefeat

KEYS = ["person_id", "index_date"]
DEMO = ["demo_age", "demo_female"]
UTIL = ["util_n_codes", "util_n_rows", "util_n_visit_days", "util_days_since_first_dx", "util_index_year"]
TEMP = ["temp_min_days_since_last", "temp_mean_days_since_last", "temp_share_recent"]
NODE_TEMP = ["node_log_days_since_last", "node_log_days_since_first", "node_recent"]
ROW_BRIDGE = ["bridge_n_flag", "bridge_sum_score", "bridge_max_score"]
NODE_GROUPS = ("STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY")
WINDOWS = {365: pl.col("in_lb365"), 1095: pl.col("in_lb1095"), 0: pl.lit(True)}


def node_table(node_features: pl.DataFrame, bridges: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    """Unscaled node features in the same category and column order as the GNN cross-check."""
    index, matrix, groups = nodefeat.build(node_features, bridges, targets, standardise=False)
    table = index.select("category").hstack(pl.DataFrame(matrix.astype(np.float64),
                                                       schema=groups["node_columns"], orient="row"))
    # Keep the original bridge-score precision for summary covariates.
    return table.update(nodefeat.bridge_table(node_features, bridges, targets), on="category").sort("category")


def _fit_scaler(table: pl.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    means = table.select(cols).mean().row(0)
    sds = table.select(cols).std(ddof=0).row(0)
    return {c: (float(mu), float(sd or 1.0)) for c, mu, sd in zip(cols, means, sds)}


def _apply_scaler(table: pl.DataFrame, scaler: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return table.with_columns(*[((pl.col(c) - mean) / sd).alias(c) for c, (mean, sd) in scaler.items()])


def fit_node_scaler(table: pl.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    """Fit over the frozen development graph's nodes."""
    return _fit_scaler(table, cols) if cols else {}


def apply_node_scaler(table: pl.DataFrame, scaler: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return _apply_scaler(table, scaler)


def fit_row_scaler(table: pl.DataFrame, cols: list[str], dev_mask: pl.Series) -> dict[str, tuple[float, float]]:
    dev = table.filter(dev_mask)
    assert dev.height and dev["split"].is_in(["train", "val"]).all(), "scaler needs development rows only"
    return _fit_scaler(dev, cols) if cols else {}


def apply_row_scaler(table: pl.DataFrame, scaler: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return _apply_scaler(table, scaler)


def write_node_artifacts(table: pl.DataFrame, out_dir: Path, row_groups: dict | None = None) -> None:
    table = table.sort("category")
    columns = [c for c in table.columns if c != "category"]
    groups = {
        "STRUCT": [c for c in columns if c in nodefeat.STRUCT_COLS],
        "BRIDGE": [c for c in columns if c.startswith("bridge_")],
        "CHAPTER": [c for c in columns if c.startswith("chapter_")],
        "COMMUNITY": [c for c in columns if c.startswith("community_")],
        "node_columns": columns,
    }
    groups.update(row_groups or row_feature_groups(table, {}))
    index = table.select("category").with_row_index("idx").select("category", pl.col("idx").cast(pl.Int32))
    nodefeat.write(index, table.select(columns).to_numpy(), groups, out_dir)


def row_feature_groups(node_table: pl.DataFrame, vocabs: dict[str, list[str]]) -> dict:
    pool_sources = [c for c in node_table.columns if c != "category"] + NODE_TEMP
    pooled = [f"pool_{stat}_{c}" for stat in ("mean", "max") for c in pool_sources]
    return {
        "CODE": [f"c_{c}" for c in node_table["category"].sort()],
        "ALL": [f"all_{c}" for c in sorted(set().union(*vocabs.values()))],
        "DEMO": DEMO,
        "UTIL": UTIL,
        "TEMP": TEMP,
        "POOL": pooled,
        "ROW_BRIDGE": ROW_BRIDGE,
        "all_by_target": {t: [f"all_{c}" for c in vocab] for t, vocab in vocabs.items()},
    }


def _bag(codes: pl.DataFrame, vocabulary: list[str], prefix: str) -> pl.DataFrame:
    columns = [f"{prefix}{c}" for c in vocabulary]
    data = codes.filter(pl.col("category").is_in(vocabulary)).with_columns(
        (pl.lit(prefix) + pl.col("category")).alias("feature"), pl.lit(1.0).alias("present"))
    wide = data.pivot(on="feature", index=KEYS, values="present", aggregate_function="first")
    return wide.with_columns([pl.lit(0.0).alias(c) for c in columns if c not in wide.columns])


def tabular_for_target(
    rows: pl.DataFrame, row_codes: pl.DataFrame, node_table_scaled: pl.DataFrame,
    target: str, cfg: dict, vocab_all: list[str] | None, lookback: int = 1095, *,
    raw_nodes: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    contract.validate(rows, "rows")
    contract.validate(row_codes, "row_codes")
    if lookback not in WINDOWS:
        raise ValueError(f"unsupported lookback {lookback}")
    pool_cols = [c for c in node_table_scaled.columns if c != "category"] + NODE_TEMP
    cats = node_table_scaled["category"].sort().to_list()
    main = rows.filter(pl.col("target") == target)
    unmasked = row_codes.filter(~pl.col(f"masked_{target}")).join(main.select(*KEYS, "split"),
                                                                on=KEYS, how="inner")
    active = unmasked.filter(WINDOWS[lookback]).drop("split")
    masked = set(row_codes.filter(pl.col(f"masked_{target}"))["category"])
    if vocab_all is not None and masked.intersection(vocab_all):
        raise ValueError(f"the supplied vocabulary contains masked categories for {target}")
    if vocab_all is None:
        vocab_all = (unmasked.filter(pl.col("split").is_in(cfg["split"]["dev"]) & WINDOWS[lookback])
                     .group_by("category").len().filter(pl.col("len") >= 10).sort("category")["category"].to_list())
    graph_codes = active.filter(pl.col("in_main_graph"))
    graph_features = graph_codes.join(node_table_scaled, on="category", how="left", validate="m:1")
    raw_nodes = node_table_scaled if raw_nodes is None else raw_nodes
    flag, score = f"bridge_flag_{target}", f"bridge_score_{target}"
    graph_features = graph_features.join(
        raw_nodes.select("category", pl.col(flag).alias("_flag"), pl.col(score).alias("_score")),
        on="category", how="left", validate="m:1",
    )
    graph_features = graph_features.with_columns(
        pl.col("days_since_last").log1p().alias(NODE_TEMP[0]),
        pl.col("days_since_first").log1p().alias(NODE_TEMP[1]),
        (pl.col("days_since_last") <= cfg["windows"]["recent_days"]).cast(pl.Float64).alias(NODE_TEMP[2]),
    )
    rec_counts = unmasked.group_by(KEYS).agg(pl.col("n_rec_lb1095").sum().alias("util_n_rows"))
    agg = graph_features.group_by(KEYS).agg(
        pl.len().alias("util_n_codes"),
        pl.col("days_since_last").min().alias(TEMP[0]),
        pl.col("days_since_last").mean().alias(TEMP[1]),
        (pl.col("days_since_last") <= cfg["windows"]["recent_days"]).mean().alias(TEMP[2]),
        pl.col("_flag").sum().alias(ROW_BRIDGE[0]),
        pl.col("_score").sum().alias(ROW_BRIDGE[1]),
        pl.col("_score").max().alias(ROW_BRIDGE[2]),
        *[getattr(pl.col(c), stat)().alias(f"pool_{stat}_{c}") for stat in ("mean", "max") for c in pool_cols],
    )
    base = list(contract.SCHEMAS["tabular"])
    groups = row_feature_groups(node_table_scaled, {target: vocab_all})
    order = base + groups["CODE"] + groups["ALL"] + DEMO + UTIL + TEMP + ROW_BRIDGE + groups["POOL"]
    table = (main.select(
        *base, pl.col("age_at_index").alias(DEMO[0]), (pl.col("sex") == "female").alias(DEMO[1]),
        *[pl.col(src).alias(dst) for src, dst in zip(
            ("n_visit_days_365", "days_since_first_dx", "index_year"), UTIL[2:])],
    ).join(agg, on=KEYS, how="left").join(rec_counts, on=KEYS, how="left")
     .join(_bag(graph_codes, cats, "c_"), on=KEYS, how="left")
     .join(_bag(active, vocab_all, "all_"), on=KEYS, how="left")
     .select(*base, *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in order[len(base):]])
     .sort(KEYS))
    return contract.validate(table, "tabular"), vocab_all


def write_graph_artifacts(p1_outputs: Path, p1_cfg: dict, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for source, name in (("graph_nodes.csv", "nodes.csv"), ("graph_edges.csv", "graph_edges.csv")):
        shutil.copyfile(p1_outputs / source, out_dir / name)
    edges = p1_cfg["edges"]
    nodes = p1_cfg["nodes"]
    thresholds = {
        "edge_rule": {k: edges[k] for k in ("cooccurrence", "test", "correction", "alpha",
                                           "min_odds_ratio", "min_shared_patients", "weight")},
        "node_rule": {k: nodes[k] for k in ("level", "min_prevalence", "main_graph_excludes_chapters")},
        "graph_patient_splits": p1_cfg["split"]["main_graph_patients"],
        "windows": cfg["windows"], "masking": cfg["masking"],
        "chapter_blocks": p1_cfg["centrality"]["chapter_blocks"],
    }
    (out_dir / "construction_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")


def build_tables(
    task: pl.DataFrame, codes: pl.DataFrame, p1_nodes: pl.DataFrame, bridges: pl.DataFrame,
    cfg: dict, out_dir: Path, *, verbose: bool = True,
) -> dict[str, list[str]]:
    """Write model-ready row tables and the shared, cross-checked node representation."""
    targets = cfg["task"]["primary_targets"] + cfg["task"]["secondary_targets"]
    # P1 bridge columns follow the canonical target order shared by the step 06 fallback.
    idx, matrix, groups = nodefeat.build(p1_nodes, bridges, contract.TARGETS)
    nt = idx.select("category").hstack(pl.DataFrame(matrix, schema=groups["node_columns"], orient="row"))
    raw_bridges = nodefeat.bridge_table(p1_nodes, bridges, contract.TARGETS)
    vocabs = {}
    for target in targets:
        table, vocabs[target] = tabular_for_target(task, codes, nt, target, cfg, None,
                                                  cfg["windows"]["primary_lookback"], raw_nodes=raw_bridges)
        counts = {}
        for split in (*cfg["split"]["dev"], *cfg["split"]["test"]):
            sub = table.filter(pl.col("split") == split)
            counts[split] = int(sub["positive"].sum())
            contract.write(sub, out_dir / f"tabular_{target}_{split}.parquet", "tabular")
        empty_share = (table["util_n_codes"] == 0).mean() or 0.0
        if verbose:
            print(f"{target}: {table.shape}, {len(vocabs[target])} all-code columns, "
              f"positives {counts}, empty code bags {empty_share:.2%}")
    write_node_artifacts(nt, out_dir, row_feature_groups(nt, vocabs))
    return vocabs


def build_synthetic(cfg: dict, out_dir: Path, *, verbose: bool = True) -> None:
    from . import inputs, rows, synth

    data = synth.make(0)
    labels = inputs.task_labels(data.labels, cfg)
    masks = rows.mask_sets(data.target_defs, cfg)
    codes = rows.build_row_codes(labels.select(KEYS).unique(), data.codes_long, data.nodes, masks, cfg)
    task = rows.build_rows(labels, codes, data.visits, data.demo, data.codes_long, cfg)
    survivors = data.bridges.group_by("category").agg(pl.col("bridge_pass").any().alias("bridge_survivor"))
    rng = np.random.default_rng(0)
    nodes = data.nodes.join(survivors, on="category").with_columns(
        *[pl.Series(c, rng.uniform(size=data.nodes.height)) for c in nodefeat.STRUCT_COLS],
        pl.Series("community_seed0", np.arange(data.nodes.height) % 3),
    )
    build_tables(task, codes, nodes, data.bridges, cfg, out_dir, verbose=verbose)
