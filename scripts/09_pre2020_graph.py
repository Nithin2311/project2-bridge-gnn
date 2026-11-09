"""Step 9: time-respecting rebuild (development records before the cutoff) and the runs that use it.

Writes outputs/pre2020/{edges.csv, target_definitions.csv, bridges.csv, node_features.csv, features/ (nodefeat
format), summary.csv, graph.graphml (written last, as the completion marker)}; then calls step 06 per primary target:
  B5-U : Layer U,        rows indexed >= cutoff
  B5-P : pre-2020 graph, rows indexed >= cutoff            (same rows, time-respecting graph and features)
  BT-U : Layer U,        dev rows <= backtest_train_to, test rows >= backtest_test_from
  BT-P : pre-2020 graph, same windows                      (the temporal backtest proper)
Rerunnable: `uv run python scripts/09_pre2020_graph.py [--skip-runs] [--arch sage] [--targets ...]`. ~15 min.
Exit code is non-zero if any step 06 run failed.
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
cutoff, era_start = str(R["cutoff"]), str(cfg["windows"]["era_start"])

if not (out / "graph.graphml").exists():
    split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
    dev_ids = split.filter(pl.col("split").is_in(cfg["split"]["dev"]))["person_id"].to_list()
    test_ids = set(split.filter(pl.col("split") == "test")["person_id"].to_list())
    con = db.connect(cfg)
    db.register_category_map(con)
    codes = extract.codes_long(con, dev_ids, era_start)
    pc = rebuild.restricted_patient_category(codes, split, cfg["split"]["dev"], era_start, cutoff)
    print(f"restricted table: {pc.height:,} (person, category) rows, {pc['person_id'].n_unique():,} dev patients before {cutoff}  ({time.time()-t0:.0f}s)", flush=True)
    nodes = pl.read_csv(p1 / "graph_nodes.csv").select("category", "name", "chapter", "flagged")
    all_categories = pl.read_csv(p1 / "nodes.csv")["category"].to_list()
    g, pairs = rebuild.build_graph(pc, nodes, p1_cfg, all_categories=all_categories, test_ids=test_ids, cutoff=cutoff)
    pairs.write_csv(out / "edges.csv")
    print(f"pre-cutoff graph: {g.number_of_nodes()} nodes, {g.number_of_edges():,} edges; Bonferroni family {int(pairs['n_tests'][0]):,} pairs  ({time.time()-t0:.0f}s)", flush=True)
    anchors = p1_cfg["centrality"]["target_clusters"]
    partition, target_defs, bridges = rebuild.cores_and_bridges(g, p1_cfg, anchors)
    target_defs.write_csv(out / "target_definitions.csv")
    bridges.write_csv(out / "bridges.csv")
    print("cores: " + ", ".join(f"{t} {target_defs.filter((pl.col('target') == t) & pl.col('in_core')).height}" for t in anchors)
          + f"; bridge survivors: {bridges.filter(pl.col('bridge_pass'))['category'].n_unique()} nodes over {bridges['pair'].n_unique()} pairs  ({time.time()-t0:.0f}s)", flush=True)
    ipid = int(p1_cfg["project2_tables"]["inpatient_visit_concept_id"])
    con.register("_dev", pl.DataFrame({"person_id": pl.Series(dev_ids, dtype=pl.Int64)}).to_arrow())
    con.register("_nodes", pl.DataFrame({"category": nodes["category"]}).to_arrow())
    inpatient = con.sql(f"""
      with ipv as (select person_id, visit_start_DATE s, coalesce(visit_end_DATE, visit_start_DATE) e from visit_occurrence where visit_concept_id = {ipid}),
      rows as (select co.person_id, m.category, co.condition_start_DATE dt from condition_occurrence co join cat_map m on m.sv = co.condition_source_value
               left join death d on d.person_id = co.person_id
               where co.condition_start_DATE >= date '{era_start}' and co.condition_start_DATE < date '{cutoff}'
                 and m.category in (select category from _nodes) and co.person_id in (select person_id from _dev)
                 and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)),
      flagged as (select r.category, exists(select 1 from ipv where ipv.person_id = r.person_id and r.dt between ipv.s and ipv.e) as inpt from rows r)
      select category, avg(case when inpt then 1 else 0 end) inpatient_share from flagged group by 1""").pl()
    demo = extract.demo(con, dev_ids)
    p1_nf = pl.read_csv(p1 / "p2" / "node_features.csv")
    nf = rebuild.node_features(g, partition, bridges, p1_nf, pc=pc, demo=demo, inpatient=inpatient, p1_cfg=p1_cfg,
                               backbone_k=int(p1_cfg["project2_tables"]["backbone_top_k"]))
    nf.write_csv(out / "node_features.csv")
    idx, X, groups = nodefeat.build(nf, bridges, contract.TARGETS)
    nodefeat.write(idx, X, groups, out / "features")
    print(f"node features recomputed on the restricted records ({X.shape[1]} columns); copied from P1: {rebuild.COPIED_FROM_P1}  ({time.time()-t0:.0f}s)", flush=True)
    g_u = nx.read_graphml(p1 / "graph_main.graphml")
    comm_u = pl.read_csv(p1 / "communities.csv")
    part_u = dict(zip(comm_u["category"], comm_u["leiden"]))
    summary = rebuild.compare(g, g_u, partition, part_u, bridges, pl.read_csv(p1 / "bridges.csv")) | {
        "cutoff": cutoff, "n_dev_patients": pc["person_id"].n_unique(), "bonferroni_family": int(pairs["n_tests"][0]),
        "empty_cores": ",".join(t for t in anchors if target_defs.filter((pl.col("target") == t) & pl.col("in_core")).height == 0) or "none"}
    pl.DataFrame([summary]).write_csv(out / "summary.csv")
    print("summary:", summary, flush=True)
    nx.write_graphml(g, out / "graph.graphml")
else:
    print(f"pre-2020 artifacts exist in {out}; skipping the rebuild (delete the folder to redo)")

if args.skip_runs:
    sys.exit(0)
arch = args.arch or yaml.safe_load((config.REPO_ROOT / "config" / "experiments.yaml").read_text()).get("default_arch", "sage")
targets = args.targets or cfg["task"]["primary_targets"]
P = ["--layer-file", str(out / "graph.graphml"), "--node-features-dir", str(out / "features")]
runs = {
    "B5-U": ["--index-from", cutoff],
    "B5-P": P + ["--index-from", cutoff],
    "BT-U": ["--index-to", str(R["backtest_train_to"]), "--test-index-from", str(R["backtest_test_from"])],
    "BT-P": P + ["--index-to", str(R["backtest_train_to"]), "--test-index-from", str(R["backtest_test_from"])],
}
failed = []
for target in targets:
    for run_id, extra in runs.items():
        cmd = [sys.executable, str(config.REPO_ROOT / "scripts" / "06_gnn.py"), "--run-id", run_id, "--target", target, "--arch", arch, *extra]
        print(f"=== {run_id} {target} ({time.time()-t0:.0f}s)", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in r.stderr.splitlines():
            if "dev rows" in line:
                print("   ", line.strip(), flush=True)
        for line in r.stdout.splitlines():
            print(line, flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            print(f"FAILED {run_id} {target}", flush=True)
            failed.append((run_id, target))
print(f"done ({time.time()-t0:.0f}s); failed runs: {failed or 'none'}")
sys.exit(1 if failed else 0)
