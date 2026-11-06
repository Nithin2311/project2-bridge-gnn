"""Step 1: task rows and per-row code sets.

Reads Project 1's p2_labels.parquet (filtered to the task), target_definitions.csv, nodes.csv, patient_split.csv
and the EHRShot CSVs through DuckDB; writes outputs/rows.parquet and outputs/row_codes.parquet (contract
`rows`, `row_codes`). Rerunnable: `uv run python scripts/01_rows.py`. A few minutes.
"""
from __future__ import annotations

import sys
import time

import polars as pl

from bridgegnn import config, contract, db, extract, inputs, rows

t0 = time.time()
cfg = config.load()
p1 = config.p1_outputs(cfg)
out = config.outputs_dir(cfg)

labels = inputs.task_labels(pl.read_parquet(p1 / "p2_labels.parquet"), cfg)
split = pl.read_csv(p1 / "patient_split.csv").select("person_id", "split")
bad = labels.join(split, on="person_id", how="left", suffix="_file").filter(pl.col("split") != pl.col("split_file"))
if bad.height:
    print(f"REFUSING: {bad.height} label rows disagree with patient_split.csv on the split")
    sys.exit(1)
nodes = pl.read_csv(p1 / "nodes.csv")
target_defs = pl.read_csv(p1 / "target_definitions.csv")
masks = rows.mask_sets(target_defs, cfg)
person_ids = sorted(labels["person_id"].unique().to_list())
print(f"task rows {labels.height:,} over {len(person_ids):,} patients; masks: " + ", ".join(f"{k} {len(v)}" for k, v in masks.items()))

con = db.connect(cfg)
db.register_category_map(con)
codes = extract.codes_long(con, person_ids, str(cfg["windows"]["era_start"]))
vis = extract.visits(con, person_ids)
dem = extract.demo(con, person_ids)
print(f"extracted {codes.height:,} (person, category, date) rows, {vis.height:,} visit days, {dem.height:,} persons  ({time.time()-t0:.0f}s)")

idx = labels.select("person_id", "index_date").unique()
rc = rows.build_row_codes(idx, codes, nodes, masks, cfg)
r = rows.build_rows(labels, rc, vis, dem, codes, cfg)
contract.write(r, out / "rows.parquet", "rows")
contract.write(rc, out / "row_codes.parquet", "row_codes")

with pl.Config(tbl_rows=20):
    print(inputs.class_balance(r.rename({"positive": "positive"})))
for lb in ["lb365", "lb1095", "lball"]:
    n = r[f"n_codes_{lb}"]
    print(f"  {lb:7s} main-graph codes per row: median {n.median():.0f} (IQR {n.quantile(0.25):.0f}-{n.quantile(0.75):.0f}); zero {(n == 0).mean():.1%}; <3 {(n < 3).mean():.1%}")
masked_share = {t: rc.filter(pl.col("in_main_graph"))[f"masked_{t}"].mean() for t in masks}
print("  share of main-graph code rows masked per target: " + ", ".join(f"{k} {v:.1%}" for k, v in masked_share.items()))
print(f"wrote rows.parquet ({r.height:,}) and row_codes.parquet ({rc.height:,})  ({time.time()-t0:.0f}s)")
