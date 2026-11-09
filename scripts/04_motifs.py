"""Step 4: triad census of Layer D against two directed nulls (free, and restricted to Layer U pairs); bridge
middle shares; A -> bridge -> core chains from first-record dates (recorded order, with an entry variant).

Inputs: outputs/layers/graph_directed.graphml (step 03), Project 1 graph_main.graphml, patient_category.parquet,
patient_split.csv, target_definitions.csv, p2/node_features.csv. Writes outputs/motifs/{triad_census.csv,
bridge_middle_share.csv, bridge_chains.csv, fig_motif_z.pdf}. Rerunnable: `uv run python scripts/04_motifs.py`. ~7 min.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl

from bridgegnn import config, directed, motifs

t0 = time.time()
cfg = config.load()
M, L = cfg["motifs"], cfg["layers"]
p1 = config.p1_outputs(cfg)
out = config.outputs_dir(cfg) / "motifs"
out.mkdir(parents=True, exist_ok=True)

dg = nx.read_graphml(config.outputs_dir(cfg) / "layers" / "graph_directed.graphml")
g_u = nx.read_graphml(p1 / "graph_main.graphml")
allowed = {frozenset(e) for e in g_u.edges()}
td = pl.read_csv(p1 / "target_definitions.csv")
cores = {t: set(td.filter((pl.col("target") == t) & pl.col("in_core"))["category"]) for t in td["target"].unique()}
nf = pl.read_csv(p1 / "p2" / "node_features.csv")
bridges = set(nf.filter(pl.col(cfg["bridge"]["survivor_flag"]))["category"])
print(f"Layer D: {dg.number_of_nodes()} nodes, {dg.number_of_edges():,} edges, DAG={nx.is_directed_acyclic_graph(dg)}; {len(bridges)} bridge codes")

census = motifs.null_census(dg, int(M["null_n"]), int(M["seed"]), allowed_pairs=allowed)
census.write_csv(out / "triad_census.csv")
with pl.Config(tbl_rows=20, tbl_width_chars=160):
    print(census.filter(pl.col("triad").is_in(list(motifs.CONNECTED))))
print(f"  census with {M['null_n']} free and {M['null_n']} U-restricted nulls  ({time.time()-t0:.0f}s)")

mid = motifs.feed_forward_middle_share(dg, bridges, cores).sort("n_middle", descending=True)
mid.write_csv(out / "bridge_middle_share.csv")
with pl.Config(tbl_width_chars=160):
    print(mid)

pc = pl.read_parquet(p1 / "patient_category.parquet")
split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
fd = directed.first_dates(pc, split, cfg["split"]["dev"], bool(L["drop_day_one_codes"]))
fd_full = directed.first_dates(pc, split, cfg["split"]["dev"], False)
chains = motifs.bridge_chains(fd, bridges, cores, int(L["precedence_gap_days"]), int(M["min_chain_patients"]),
                              fd_full=fd_full, node_set=set(g_u.nodes))
chains.write_csv(out / "bridge_chains.csv")
with pl.Config(tbl_rows=int(M["top_chains"]), tbl_width_chars=160):
    print(chains.head(int(M["top_chains"])))
n_graph = chains.filter(pl.col("a_in_graph")).height
print(f"  {chains.height:,} chains with >= {M['min_chain_patients']} patients; {n_graph:,} start from a main-graph code, "
      f"{chains.height - n_graph:,} from R/Z/O or sub-floor codes; entry-chain patients are the n_patients_entry column  ({time.time()-t0:.0f}s)")

conn = census.filter(pl.col("triad").is_in(list(motifs.CONNECTED)))
x = np.arange(conn.height)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(x - 0.2, [v if v is not None else 0 for v in conn["z"].to_list()], 0.4, label="free directed rewiring", color="#AAAAAA")
ax.bar(x + 0.2, [v if v is not None else 0 for v in conn["z_u"].to_list()], 0.4, label="rewiring on Layer U pairs", color="#4C72B0")
ax.set_xticks(x, conn["triad"].to_list())
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel(f"z vs. {M['null_n']} rewirings")
ax.set_xlabel("triad class (NetworkX); Layer D is a DAG, so classes with reciprocal edges or cycles are 0 by construction")
ax.set_title("Layer D triad census")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "fig_motif_z.pdf")
fig.savefig(out / "fig_motif_z.png", dpi=150)
print(f"done ({time.time()-t0:.0f}s)")
