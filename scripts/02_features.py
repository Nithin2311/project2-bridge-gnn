"""Build node, graph and masked tabular feature artifacts (step 02)."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import polars as pl

from bridgegnn import config, contract, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--p1-outputs", type=Path, help="override the read-only Project 1 handoff directory")
    parser.add_argument("--rows-dir", type=Path, help="override the directory containing step 01 parquets")
    parser.add_argument("--outputs", type=Path, help="override the Project 2 output directory")
    args = parser.parse_args()
    cfg = config.load()
    if args.synthetic:
        if args.outputs:
            features.build_synthetic(cfg, args.outputs / "features")
        else:
            with tempfile.TemporaryDirectory(prefix="bridgegnn-features-") as tmp:
                features.build_synthetic(cfg, Path(tmp) / "features")
        return

    p1 = args.p1_outputs or config.p1_outputs(cfg)
    out = args.outputs or config.outputs_dir(cfg)
    rows_dir = args.rows_dir or out
    task = contract.read(rows_dir / "rows.parquet", "rows")
    codes = contract.read(rows_dir / "row_codes.parquet", "row_codes")
    nodes = pl.read_csv(p1 / "p2" / "node_features.csv")
    graph_nodes = pl.read_csv(p1 / "graph_nodes.csv", columns=["category"])
    if not nodes["category"].is_unique().all() or set(nodes["category"]) != set(graph_nodes["category"]):
        raise ValueError("Project 1 node features must cover the main graph exactly")
    if codes.filter(pl.col("in_main_graph") != pl.col("category").is_in(nodes["category"].to_list())).height:
        raise ValueError("Step 01 rows and Project 1 main graph disagree on node membership")
    features.build_tables(task, codes, nodes, pl.read_csv(p1 / "bridges.csv"), cfg, out / "features")
    artifacts = out / "artifacts"
    features.write_graph_artifacts(p1, config.p1_config(cfg), artifacts, cfg)
    for name in ("node_index.csv", "node_x.npy", "feature_groups.json"):
        shutil.copyfile(out / "features" / name, artifacts / name)


if __name__ == "__main__":
    main()
