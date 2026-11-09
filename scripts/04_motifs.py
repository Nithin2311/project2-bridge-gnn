"""Step 4: triad census of Layer D against directed rewired nulls; bridge-centred chains from first-record dates.

Inputs: outputs/layers/graph_directed.graphml (step 03), Project 1 patient_category.parquet, patient_split.csv,
target_definitions.csv, p2/node_features.csv. Writes outputs/motifs/{triad_census.csv, bridge_middle_share.csv,
bridge_chains.csv, fig_motif_z.pdf}. Rerunnable: `uv run python scripts/04_motifs.py`.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import polars as pl  # noqa: E402

from bridgegnn import config, directed, motifs  # noqa: E402

t0 = time.time()
cfg = config.load()
M, L = cfg["motifs"], cfg["layers"]
p1 = config.p1_outputs(cfg)
out = config.outputs_dir(cfg) / "motifs"
out.mkdir(parents=True, exist_ok=True)

dg = nx.read_graphml(config.outputs_dir(cfg) / "layers" / "graph_directed.graphml")
td = pl.read_csv(p1 / "target_definitions.csv")
cores = {t: set(td.filter((pl.col("target") == t) & pl.col("in_core"))["category"]) for t in td["target"].unique()}
nf = pl.read_csv(p1 / "p2" / "node_features.csv")
bridges = set(nf.filter(pl.col(cfg["bridge"]["survivor_flag"]))["category"])
print(f"Layer D: {dg.number_of_nodes()} nodes, {dg.number_of_edges():,} edges; {len(bridges)} bridge codes")

census = motifs.null_census(dg, int(M["null_n"]), int(M["seed"]))
census.write_csv(out / "triad_census.csv")
with pl.Config(tbl_rows=20):
    print(census.filter(pl.col("triad").is_in(list(motifs.CONNECTED))))
print(f"  census with {M['null_n']} nulls  ({time.time()-t0:.0f}s)")

mid = motifs.feed_forward_middle_share(dg, bridges).sort("n_middle", descending=True)
mid.write_csv(out / "bridge_middle_share.csv")
print(mid)

pc = pl.read_parquet(p1 / "patient_category.parquet")
split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
fd = directed.first_dates(pc, split, cfg["split"]["dev"], bool(L["drop_day_one_codes"]))
chains = motifs.bridge_chains(fd, bridges, cores, int(L["precedence_gap_days"]), int(M["min_chain_patients"]))
chains.write_csv(out / "bridge_chains.csv")
with pl.Config(tbl_rows=int(M["top_chains"])):
    print(chains.head(int(M["top_chains"])))
print(f"  {chains.height:,} chains with >= {M['min_chain_patients']} patients  ({time.time()-t0:.0f}s)")

conn = census.filter(pl.col("triad").is_in(list(motifs.CONNECTED)) & pl.col("z").is_not_null())
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(conn["triad"].to_list(), conn["z"].to_list(), color="#4C72B0")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("z vs. 100 directed rewirings")
ax.set_xlabel("triad class (NetworkX)")
ax.set_title("Layer D triad census")
fig.tight_layout()
fig.savefig(out / "fig_motif_z.pdf")
fig.savefig(out / "fig_motif_z.png", dpi=150)
print(f"done ({time.time()-t0:.0f}s)")
