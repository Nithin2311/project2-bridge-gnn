"""Step 3: Layer D (directed precedence edges) with sensitivities, the U-bh edge variant, and the null graphs.

Inputs (Project 1): graph_main.graphml, patient_category.parquet, patient_split.csv, edges.csv (all tested pairs),
target_definitions.csv, p2/node_features.csv, config/option_a.yaml (edge floors). Development patients only.
Writes outputs/layers/{edges_directed*.csv, graph_directed.graphml, edges_bh.csv, graph_bh.graphml, null/*.graphml,
tables/*.csv}. Rerunnable: `uv run python scripts/03_directed_layer.py`. About 10 s.
"""
from __future__ import annotations

import time

import networkx as nx
import polars as pl

from bridgegnn import config, contract, directed, layers

t0 = time.time()
cfg = config.load()
cfg["p1_edges"] = config.p1_config(cfg)["edges"]      # OR and support floors, read from the frozen P1 config
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
gap, drop = int(L["precedence_gap_days"]), bool(L["drop_day_one_codes"])
min_total, alpha = int(L["min_pair_patients"]), float(L["alpha"])
print(f"Layer U: {g_u.number_of_nodes()} nodes, {len(pairs):,} edges; cores " + ", ".join(f"{k} {len(v)}" for k, v in cores.items()) + f"; {len(bridges)} bridge codes")


def run(tag: str, gap_days: int, drop_day_one: bool) -> tuple[pl.DataFrame, dict]:
    fd = directed.first_dates(pc, split, cfg["split"]["dev"], drop_day_one)
    counts = directed.precedence_counts(fd, pairs, gap_days)
    n_tested = counts.filter((pl.col("n_ab") + pl.col("n_ba")) >= min_total).height
    de = directed.attach_phi(directed.directed_edges(counts, min_total, alpha), g_u)
    contract.write(de, out / f"edges_directed{tag}.csv", "edges_directed")
    s = directed.summary(de, g_u, n_tested) | {"run": tag.lstrip("_") or "primary", "gap_days": gap_days, "drop_day_one": drop_day_one,
                                               "n_dev_patients_with_retained_codes": fd["person_id"].n_unique()}
    print(f"  Layer D {s['run']:12s} gap {gap_days:3d} d drop_day_one={drop_day_one!s:5s}: tested {n_tested:,}, directed {s['n_directed']:,} "
          f"({s['share_directed']:.1%} of U), undetermined {s['share_undetermined']:.1%}, median share {s['median_share']:.2f}  ({time.time()-t0:.0f}s)")
    return de, s


# ---- Layer D primary, then sensitivities ----
de_primary, s_primary = run("", gap, drop)
dg = directed.to_digraph(de_primary, g_u)
s_primary["is_dag"] = nx.is_directed_acyclic_graph(dg)
nx.write_graphml(dg, out / "graph_directed.graphml")
directed.edges_into_cores(de_primary, cores, int(L["top_k_into_cores"])).write_csv(out / "tables" / "directed_into_cores.csv")
directed.edges_through_bridges(de_primary, bridges).write_csv(out / "tables" / "directed_through_bridges.csv")
summaries = [s_primary]
for gday in L["gap_sensitivity_days"]:
    summaries.append(run(f"_g{int(gday)}", int(gday), drop)[1])
summaries.append(run("_incl_dayone", gap, False)[1])
pl.DataFrame(summaries).write_csv(out / "tables" / "directed_summary.csv")
print(f"  Layer D is a DAG: {s_primary['is_dag']}")

# ---- U-bh ----
edges_all = pl.read_csv(p1 / "edges.csv")
bh, family_n = layers.bh_edges(edges_all, set(g_u.nodes), cfg)
bh.write_csv(out / "edges_bh.csv")
g_bh = layers.graph_from_edges(bh, g_u)
nx.write_graphml(g_bh, out / "graph_bh.graphml")
print(f"  U-bh: {bh.height:,} edges (BH over the P1 family of {family_n:,} tested pairs; Bonferroni main graph {g_u.number_of_edges():,})")

# ---- Layer N and stats ----
stats = [{"layer": "U"} | layers.layer_stats(g_u), {"layer": "U-bh", "bh_family_n": family_n} | layers.layer_stats(g_bh),
         {"layer": "D"} | layers.layer_stats(dg)]
g_bb = nx.read_graphml(p1 / "p2" / "graph_backbone_topk.graphml")
stats.append({"layer": "U-bb"} | layers.layer_stats(g_bb))
for i, h in enumerate(layers.rewired_graphs(g_u, int(L["null_n"]), int(L["null_seed"]))):
    nx.write_graphml(h, out / "null" / f"rewire{i}.graphml")
    stats.append({"layer": f"N-rewire{i}"} | layers.layer_stats(h))
sh = layers.shuffled_graph(g_u, int(L["null_seed"]))
nx.write_graphml(sh, out / "null" / "shuffle.graphml")
stats.append({"layer": "N-shuffle"} | layers.layer_stats(sh))
pl.DataFrame(stats, infer_schema_length=None).write_csv(out / "tables" / "layer_stats.csv")

with pl.Config(tbl_rows=30, tbl_width_chars=140):
    print(pl.DataFrame(stats, infer_schema_length=None))
    print(pl.read_csv(out / "tables" / "directed_into_cores.csv").head(20))
print(f"done ({time.time()-t0:.0f}s)")
