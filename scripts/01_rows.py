"""Step 1: task rows and per-row code sets.

Reads Project 1's p2_labels.parquet (filtered to the task), target_definitions.csv, graph_nodes.csv (the 496-node
main graph), patient_split.csv, the P1 config's chapter_blocks ranges, and the EHRShot CSVs through DuckDB; writes
outputs/rows.parquet and outputs/row_codes.parquet (contract `rows`, `row_codes`).

Hard checks against Project 1 before writing: every label row keeps its split; days_since_first_dx equals the
label's history_days; unmasked main-graph code counts per (patient, index) equal P1's patient_index_subgraphs.csv.
Rerunnable: `uv run python scripts/01_rows.py`. About 2.5 minutes.
"""
from __future__ import annotations

import sys
import time

import polars as pl

from bridgegnn import config, contract, db, extract, inputs, rows


def fail(msg: str) -> None:
    print(f"REFUSING: {msg}")
    sys.exit(1)


t0 = time.time()
cfg = config.load()
p1 = config.p1_outputs(cfg)
p1_cfg = config.p1_config(cfg)
out = config.outputs_dir(cfg)

labels_full = pl.read_parquet(p1 / "p2_labels.parquet")
labels = inputs.task_labels(labels_full, cfg)
split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
chk = labels.join(split, on="person_id", how="left", suffix="_file")
bad = chk.filter(pl.col("split_file").is_null() | (pl.col("split") != pl.col("split_file")))
if bad.height:
    fail(f"{bad.height} label rows disagree with patient_split.csv on the split")
graph_nodes = pl.read_csv(p1 / "graph_nodes.csv")
if graph_nodes.height != 496:
    fail(f"graph_nodes.csv has {graph_nodes.height} rows, expected the 496-node main graph")
target_defs = pl.read_csv(p1 / "target_definitions.csv")
person_ids = sorted(labels["person_id"].unique().to_list())
print(f"task rows {labels.height:,} over {len(person_ids):,} patients")

con = db.connect(cfg)
db.register_category_map(con)
codes = extract.codes_long(con, person_ids, str(cfg["windows"]["era_start"]))
vis = extract.visits(con, person_ids)
dem = extract.demo(con, person_ids)
print(f"extracted {codes.height:,} (person, category, date) rows, {vis.height:,} visit days, {dem.height:,} persons  ({time.time()-t0:.0f}s)")

masks = rows.mask_sets(target_defs, cfg, chapter_blocks=p1_cfg["centrality"]["chapter_blocks"],
                       categories=codes["category"].unique().to_list())
if set(masks) != set(contract.TARGETS):
    fail(f"masks cover {sorted(masks)}, expected {contract.TARGETS}")
print("masks (core + chapter-block ranges over all observed categories): " + ", ".join(f"{k} {len(v)}" for k, v in masks.items()))

idx = labels.select("person_id", "index_date").unique()
rc = rows.build_row_codes(idx, codes, graph_nodes, masks, cfg)
r = rows.build_rows(labels, rc, vis, dem, codes, cfg)

# ---- checks against Project 1 ----
if r.height != labels.height:
    fail(f"rows {r.height} != labels {labels.height}")
if r["age_at_index"].null_count() or r["sex"].null_count():
    fail("null age or sex")
h = r.join(labels.select("person_id", "index_date", "target", "history_days"), on=["person_id", "index_date", "target"])
if (h["days_since_first_dx"] != h["history_days"]).any():
    fail(f"days_since_first_dx != history_days for {(h['days_since_first_dx'] != h['history_days']).sum()} rows")
unmasked = rc.group_by("person_id", "index_date").agg(
    (pl.col("in_lb365") & pl.col("in_main_graph")).sum().alias("c365"),
    (pl.col("in_lb1095") & pl.col("in_main_graph")).sum().alias("c1095"),
    pl.col("in_main_graph").sum().alias("call"))
ref = pl.read_csv(p1 / "p2" / "patient_index_subgraphs.csv", try_parse_dates=True).select(
    "person_id", "index_date", "lb365_n_codes", "lb1095_n_codes", "lball_n_codes")
cmp = unmasked.join(ref, on=["person_id", "index_date"], how="inner")
if cmp.height != unmasked.height:
    fail(f"{unmasked.height - cmp.height} (patient, index) rows missing from P1 patient_index_subgraphs.csv")
mism = cmp.filter((pl.col("c365") != pl.col("lb365_n_codes")) | (pl.col("c1095") != pl.col("lb1095_n_codes")) | (pl.col("call") != pl.col("lball_n_codes")))
if mism.height:
    print(mism.head(10))
    fail(f"main-graph code counts differ from Project 1 for {mism.height} of {cmp.height} (patient, index) rows")
print(f"checks passed: splits, history_days, and P1 subgraph code counts on {cmp.height:,} (patient, index) rows")

contract.write(r, out / "rows.parquet", "rows")
contract.write(rc, out / "row_codes.parquet", "row_codes")
with pl.Config(tbl_rows=20):
    print(inputs.class_balance(r))
for lb in ["lb365", "lb1095", "lball"]:
    n = r[f"n_codes_{lb}"]
    print(f"  {lb:7s} unmasked main-graph codes per row: median {n.median():.0f} (IQR {n.quantile(0.25):.0f}-{n.quantile(0.75):.0f}); zero {(n == 0).mean():.1%}; <3 {(n < 3).mean():.1%}")
masked_share = {t: rc.filter(pl.col("in_main_graph"))[f"masked_{t}"].mean() for t in masks}
print("  share of main-graph code rows masked per target: " + ", ".join(f"{k} {v:.1%}" for k, v in masked_share.items()))
print(f"wrote rows.parquet ({r.height:,}) and row_codes.parquet ({rc.height:,})  ({time.time()-t0:.0f}s)")
