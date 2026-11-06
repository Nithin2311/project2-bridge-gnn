"""DuckDB access to the EHRShot CSVs for Project 2.

Project 2 owns its own cache file and re-creates every view on connect, so a moved data folder (which left
Project 1's cache pointing at a stale absolute path) can never break a run. Nothing is copied.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import polars as pl

from bridgedx.cohort import icd10_3char
from bridgedx.data import TABLES

from . import config


def _data_dir(cfg: dict) -> Path:
    env = os.environ.get(cfg["paths"]["data_env"])
    if env:
        return Path(env).resolve()
    p = Path(cfg["paths"]["data_default"])
    return p.resolve() if p.is_absolute() else (config.REPO_ROOT / p).resolve()


def _cache_path(cfg: dict) -> Path:
    p = Path(cfg["paths"].get("cache", ".cache/ehrshot.duckdb"))
    p = p if p.is_absolute() else config.REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(cfg: dict | None = None) -> duckdb.DuckDBPyConnection:
    """Open the Project 2 cache and (re)create a view per EHRShot CSV found in the data dir."""
    cfg = cfg or config.load()
    data = _data_dir(cfg)
    con = duckdb.connect(str(_cache_path(cfg)))
    for t in TABLES:
        csv = data / f"{t}.csv"
        if csv.exists():
            con.execute(f"create or replace view {t} as select * from read_csv_auto('{csv}', header=true, sample_size=-1)")
    return con


def register_category_map(con: duckdb.DuckDBPyConnection) -> None:
    """Temp table cat_map(sv, category): Project 1's ICD-10 source-value to 3-char category map.

    Same mapping function as Project 1 (bridgedx.cohort.icd10_3char), re-run here because P1's helper uses
    connection.pl(), removed in duckdb 2.0."""
    vals = con.sql("select distinct condition_source_value as sv from condition_occurrence "
                   "where condition_source_value is not null").pl()
    mapped = vals.with_columns(pl.col("sv").map_elements(icd10_3char, return_dtype=pl.Utf8).alias("category")).drop_nulls("category")
    con.register("_cat_map_df", mapped.to_arrow())
    con.execute("create or replace temp table cat_map as select * from _cat_map_df")
