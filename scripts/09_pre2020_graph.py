"""Step 9: time-respecting rebuild (development records before the cutoff) and the runs that use it.

Writes outputs/pre2020/{graph.graphml, edges.csv, node_features.csv, bridges.csv, target_definitions.csv,
features/ (nodefeat format), summary.csv}; then calls step 06 for, per primary target:
  B5-U  : Layer U,        rows indexed >= cutoff
  B5-P  : pre-2020 graph, rows indexed >= cutoff            (same rows, time-respecting graph and features)
  BT-U  : Layer U,        dev rows <= backtest_train_to, test rows >= backtest_test_from
  BT-P  : pre-2020 graph, same windows                      (the temporal backtest proper)
Rerunnable: `uv run python scripts/09_pre2020_graph.py [--skip-runs] [--arch sage]`. Rebuild ~10-20 min (bridge nulls).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import networkx as nx
import polars as pl
import yaml

from bridgegnn import config, contract, db, extract, nodefeat, rebuild

ap = argparse.ArgumentParser()
ap.add_argument("--skip-runs", action="store_true")
ap.add_argument("--arch", default=None)
ap.add_argument("--targets", nargs="+", default=None)
args = ap.parse_args()
t0 = time.time()
cfg = config.load()
p1_cfg = config.p1_config(cfg)
R = cfg["rebuild"]
p1 = config.p1_outputs(cfg)
out = config.outputs_dir(cfg) / "pre2020"
out.mkdir(parents=True, exist_ok=True)

if not (out / "graph.graphml").exists():
    split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
    dev_ids = split.filter(pl.col("split").is_in(cfg["split"]["dev"]))["person_id"].to_list()
    con = db.connect(cfg)
    db.register_category_map(con)
    codes = extract.codes_long(con, dev_ids, str(cfg["windows"]["era_start"]))
    pc = rebuild.restricted_patient_category(codes, split, cfg["split"]["dev"], str(cfg["windows"]["era_start"]), str(R["cutoff"]))
    print(f"restricted table: {pc.height:,} (person, category) rows, {pc['person_id'].n_unique():,} dev patients before {R['cutoff']}  ({time.time()-t0:.0f}s)", flush=True)
    nodes = pl.read_csv(p1 / "graph_nodes.csv").select("category", "name", "chapter", "flagged")
    g, pairs = rebuild.build_graph(pc, nodes, p1_cfg)
    pairs.write_csv(out / "edges.csv")
    print(f"pre-cutoff graph: {g.number_of_nodes()} nodes, {g.number_of_edges():,} edges  ({time.time()-t0:.0f}s)", flush=True)
    anchors = p1_cfg["centrality"]["target_clusters"]
    partition, target_defs, bridges = rebuild.cores_and_bridges(g, p1_cfg, anchors)
    target_defs.write_csv(out / "target_definitions.csv")
    bridges.write_csv(out / "bridges.csv")
    print("cores: " + ", ".join(f"{t} {target_defs.filter((pl.col('target') == t) & pl.col('in_core')).height}" for t in anchors)
          + f"; bridges passing: {int(bridges['bridge_pass'].sum())}  ({time.time()-t0:.0f}s)", flush=True)
    p1_nf = pl.read_csv(p1 / "p2" / "node_features.csv")
    nf = rebuild.node_features(g, partition, bridges, p1_nf)
    nf.write_csv(out / "node_features.csv")
    idx, X, groups = nodefeat.build(nf, bridges, contract.TARGETS)
    nodefeat.write(idx, X, groups, out / "features")
    nx.write_graphml(g, out / "graph.graphml")
    g_u = nx.read_graphml(p1 / "graph_main.graphml")
    comm_u = pl.read_csv(p1 / "communities.csv")
    part_u = dict(zip(comm_u["category"], comm_u["leiden"]))
    summary = rebuild.compare(g, g_u, partition, part_u, bridges, pl.read_csv(p1 / "bridges.csv")) | {"cutoff": str(R["cutoff"]), "n_dev_patients": pc["person_id"].n_unique()}
    pl.DataFrame([summary]).write_csv(out / "summary.csv")
    print("summary:", summary, flush=True)
else:
    print(f"pre-2020 artifacts exist in {out}; skipping the rebuild (delete the folder to redo)")

if args.skip_runs:
    sys.exit(0)
arch = args.arch or yaml.safe_load((config.REPO_ROOT / "config" / "experiments.yaml").read_text()).get("default_arch", "sage")
targets = args.targets or cfg["task"]["primary_targets"]
P = ["--layer-file", str(out / "graph.graphml"), "--node-features-dir", str(out / "features")]
runs = {
    "B5-U": ["--index-from", str(R["cutoff"])],
    "B5-P": P + ["--index-from", str(R["cutoff"])],
    "BT-U": ["--index-to", str(R["backtest_train_to"]), "--test-index-from", str(R["backtest_test_from"])],
    "BT-P": P + ["--index-to", str(R["backtest_train_to"]), "--test-index-from", str(R["backtest_test_from"])],
}
for target in targets:
    for run_id, extra in runs.items():
        cmd = [sys.executable, str(config.REPO_ROOT / "scripts" / "06_gnn.py"), "--run-id", run_id, "--target", target, "--arch", arch, *extra]
        print(f"=== {run_id} {target} ({time.time()-t0:.0f}s)", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            print(line, flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            print(f"FAILED {run_id} {target}", flush=True)
print(f"done ({time.time()-t0:.0f}s)")
