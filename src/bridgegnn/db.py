"""DuckDB access to the EHRShot CSVs for Project 2.

The connection is in-memory and every view is created on connect from the resolved data dir, so a moved data
folder (which left Project 1's on-disk cache pointing at a stale absolute path) or a duckdb upgrade (which made
a 2.0-written cache unreadable by 1.x) can never break a run. Nothing is copied; nothing is cached on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
from bridgedx.cohort import _register_category_map
from bridgedx.data import TABLES

from . import config


def _data_dir(cfg: dict) -> Path:
    env = os.environ.get(cfg["paths"]["data_env"])
    if env:
        return Path(env).resolve()
    p = Path(cfg["paths"]["data_default"])
    return p.resolve() if p.is_absolute() else (config.REPO_ROOT / p).resolve()


def connect(cfg: dict | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per EHRShot CSV found in the data dir; raises if none is found."""
    cfg = cfg or config.load()
    data = _data_dir(cfg)
    con = duckdb.connect()
    found = 0
    for t in TABLES:
        csv = data / f"{t}.csv"
        if csv.exists():
            con.execute(f"create or replace view {t} as select * from read_csv_auto('{csv}', header=true, sample_size=-1)")
            found += 1
    if not found:
        raise FileNotFoundError(f"no EHRShot CSVs found in {data}; set {cfg['paths']['data_env']} or paths.data_default")
    return con


def register_category_map(con: duckdb.DuckDBPyConnection) -> None:
    """Temp table cat_map(sv, category): Project 1's own ICD-10 source-value to 3-char category map
    (bridgedx.cohort._register_category_map; needs duckdb < 2, pinned in pyproject)."""
    _register_category_map(con)
