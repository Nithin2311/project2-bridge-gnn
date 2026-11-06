"""Step 0: verify Project 1 outputs, the EHRShot DuckDB cache, and print the task's class balance.

Rerunnable: `uv run python scripts/00_check_inputs.py`. Writes nothing.
"""
from __future__ import annotations

import sys

import polars as pl

from bridgegnn import config, inputs

cfg = config.load()
p1 = config.p1_outputs(cfg)
missing = inputs.check(p1)
print(f"Project 1 outputs: {p1}")
if missing:
    print("MISSING:", *missing, sep="\n  ")
    sys.exit(1)
print(f"  all {len(inputs.P1_FILES)} required files present")

from bridgegnn import db

con = db.connect(cfg)
print(f"  data dir {db._data_dir(cfg)}")
for t in ["person", "condition_occurrence", "visit_occurrence", "death"]:
    n = con.execute(f"select count(*) from {t}").fetchone()[0]
    print(f"  duckdb {t:22s} {n:>12,} rows")

labels = inputs.task_labels(pl.read_parquet(p1 / "p2_labels.parquet"), cfg)
bal = inputs.class_balance(labels)
print(f"\nTask rows: definition={cfg['task']['definition']} horizon={cfg['task']['horizon_months']} mo, "
      f"censored excluded={cfg['task']['exclude_censored']}\n")
with pl.Config(tbl_rows=40, tbl_width_chars=120):
    print(bal)
low = bal.filter((pl.col("split") == "all") & (pl.col("positives") < 150))["target"].to_list()
if low:
    print(f"\nunderpowered (< 150 positives): {low}  -> secondary targets per config: {cfg['task']['secondary_targets']}")
