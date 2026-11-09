"""Step 3: Layer D (directed precedence edges) with sensitivities, the U-bh edge variant, and the null graphs.

Inputs (Project 1): graph_main.graphml, patient_category.parquet, patient_split.csv, edges.csv (all tested pairs),
target_definitions.csv, p2/node_features.csv. Development patients only.
Writes outputs/layers/{edges_directed*.csv, graph_directed.graphml, edges_bh.csv, graph_bh.graphml, null/*.graphml,
tables/*.csv}. Rerunnable: `uv run python scripts/03_directed_layer.py`. A few minutes (rewiring 9,115 edges x 5).
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgegnn import config, contract, directed, layers

t0 = time.time()
cfg = config.load()
L = cfg["layers"]
p1 = config.p1_outputs(cfg)
out = config.outputs_dir(cfg) / "layers"
(out / "null").mkdir(parents=True, exist_ok=True)
(out / "tables").mkdir(exist_ok=True)

g_u = nx.read_graphml(p1 / "graph_main.graphml")
pc = pl.read_parquet(p1 / "patient_category.parquet")
split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
td = pl.read_csv(p1 / "target_definitions.csv")
cores = {t: set(td.filter((pl.col("target") == t) & pl.col("in_core"))["category"]) for t in td["target"].unique()}
nf = pl.read_csv(p1 / "p2" / "node_features.csv")
bridges = set(nf.filter(pl.col(cfg["bridge"]["survivor_flag"]))["category"])
pairs = [(u, v) for u, v in g_u.edges()]
print(f"Layer U: {g_u.number_of_nodes()} nodes, {len(pairs):,} edges; cores " + ", ".join(f"{k} {len(v)}" for k, v in cores.items()) + f"; {len(bridges)} bridge codes")

# ---- Layer D primary + sensitivities ----
runs = {"": (int(L["precedence_gap_days"]), bool(L["drop_day_one_codes"]))}
for gday in L["gap_sensitivity_days"]:
    runs[f"_g{int(gday)}"] = (int(gday), bool(L["drop_day_one_codes"]))
runs["_dayone"] = (int(L["precedence_gap_days"]), False)
summaries = []
for tag, (gap, drop) in runs.items():
    fd = directed.first_dates(pc, split, cfg["split"]["dev"], drop)
    counts = directed.precedence_counts(fd, pairs, gap)
    n_tested = counts.filter((pl.col("n_ab") + pl.col("n_ba")) >= int(L["min_pair_patients"])).height
    de = directed.attach_phi(directed.directed_edges(counts, int(L["min_pair_patients"]), float(L["alpha"])), g_u)
    contract.write(de, out / f"edges_directed{tag}.csv", "edges_directed")
    s = directed.summary(de, g_u, n_tested) | {"run": tag or "primary", "gap_days": gap, "drop_day_one": drop,
                                               "n_dev_patients": fd["person_id"].n_unique()}
    summaries.append(s)
    if tag == "":
        dg = directed.to_digraph(de, g_u)
        nx.write_graphml(dg, out / "graph_directed.graphml")
        directed.edges_into_cores(de, cores, int(L["top_k_into_cores"])).write_csv(out / "tables" / "directed_into_cores.csv")
        directed.edges_through_bridges(de, bridges).write_csv(out / "tables" / "directed_through_bridges.csv")
        de_primary = de
    print(f"  Layer D {s['run']:8s} gap {gap:3d} d drop_day_one={drop!s:5s}: tested {n_tested:,}, directed {s['n_directed']:,} "
          f"({s['share_directed']:.1%} of U), median share {s['median_share']:.2f}  ({time.time()-t0:.0f}s)")
pl.DataFrame(summaries).write_csv(out / "tables" / "directed_summary.csv")

# ---- U-bh ----
edges_all = pl.read_csv(p1 / "edges.csv")
bh = layers.bh_edges(edges_all, set(g_u.nodes), cfg)
bh.write_csv(out / "edges_bh.csv")
g_bh = layers.graph_from_edges(bh, g_u)
nx.write_graphml(g_bh, out / "graph_bh.graphml")
print(f"  U-bh: {bh.height:,} edges (Bonferroni main graph {g_u.number_of_edges():,})")

# ---- Layer N ----
stats = [{"layer": "U"} | layers.layer_stats(g_u), {"layer": "U-bh"} | layers.layer_stats(g_bh),
         {"layer": "D"} | layers.layer_stats(directed.to_digraph(de_primary, g_u))]
g_bb = nx.read_graphml(p1 / "p2" / "graph_backbone_topk.graphml")
stats.append({"layer": "U-bb"} | layers.layer_stats(g_bb))
for i, h in enumerate(layers.rewired_graphs(g_u, int(L["null_n"]), int(L["null_seed"]))):
    nx.write_graphml(h, out / "null" / f"rewire{i}.graphml")
    stats.append({"layer": f"N-rewire{i}"} | layers.layer_stats(h))
    print(f"  null rewire{i} written  ({time.time()-t0:.0f}s)")
sh = layers.shuffled_graph(g_u, int(L["null_seed"]))
nx.write_graphml(sh, out / "null" / "shuffle.graphml")
stats.append({"layer": "N-shuffle"} | layers.layer_stats(sh))
pl.DataFrame(stats).write_csv(out / "tables" / "layer_stats.csv")

with pl.Config(tbl_rows=30, tbl_width_chars=140):
    print(pl.DataFrame(stats))
    print(pl.read_csv(out / "tables" / "directed_into_cores.csv").head(20))
print(f"done ({time.time()-t0:.0f}s)")
