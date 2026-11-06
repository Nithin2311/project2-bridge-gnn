"""The data contract: schemas of every file that crosses between steps and people.

Validate on write and on read. See docs/DATA-CONTRACT.md for the prose version.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

TARGETS = ["cardiometabolic", "cancer_metastatic", "cancer_hematologic", "mental_health"]

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "rows": {
        "person_id": pl.Int64, "index_date": pl.Date, "target": pl.Utf8, "split": pl.Utf8,
        "positive": pl.Boolean, "days_to_event": pl.Int64,
        "age_at_index": pl.Float64, "sex": pl.Utf8, "index_year": pl.Int32, "days_since_first_dx": pl.Int32,
        "n_codes_lb365": pl.Int32, "n_codes_lb1095": pl.Int32, "n_codes_lball": pl.Int32,
        "n_rows_lb1095": pl.Int32, "n_visit_days_365": pl.Int32,
    },
    "row_codes": {
        "person_id": pl.Int64, "index_date": pl.Date, "category": pl.Utf8,
        "first_date": pl.Date, "last_date": pl.Date,
        "days_since_first": pl.Int32, "days_since_last": pl.Int32,
        "in_lb365": pl.Boolean, "in_lb1095": pl.Boolean, "in_main_graph": pl.Boolean,
        **{f"masked_{t}": pl.Boolean for t in TARGETS},
    },
    "node_index": {"category": pl.Utf8, "idx": pl.Int32},
    "tabular": {
        "person_id": pl.Int64, "index_date": pl.Date, "target": pl.Utf8, "split": pl.Utf8, "positive": pl.Boolean,
    },
    "predictions": {
        "run_id": pl.Utf8, "model": pl.Utf8, "target": pl.Utf8, "seed": pl.Int32,
        "person_id": pl.Int64, "index_date": pl.Date, "split": pl.Utf8, "y": pl.Boolean, "p": pl.Float64,
    },
    "results": {
        "run_id": pl.Utf8, "family": pl.Utf8, "model": pl.Utf8, "target": pl.Utf8, "seed": pl.Int32,
        "layer": pl.Utf8, "features": pl.Utf8, "params_json": pl.Utf8,
        "dev_oof_auprc": pl.Float64, "test_auroc": pl.Float64, "test_auprc": pl.Float64, "test_brier": pl.Float64,
        "n_test_rows": pl.Int32, "n_test_pos": pl.Int32,
    },
}

EXTRA_COLUMNS_ALLOWED = {"tabular"}
FEATURE_PREFIXES = ("c_", "all_", "demo_", "util_", "temp_", "bridge_", "pool_mean_", "pool_max_")


class ContractError(ValueError):
    pass


def validate(df: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in SCHEMAS:
        raise ContractError(f"unknown contract {name!r}; known: {sorted(SCHEMAS)}")
    sch = SCHEMAS[name]
    problems = []
    missing = [c for c in sch if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    for c, t in sch.items():
        if c in df.columns and df.schema[c] != t:
            problems.append(f"column {c!r} has dtype {df.schema[c]}, expected {t}")
    extra = [c for c in df.columns if c not in sch]
    if extra and name not in EXTRA_COLUMNS_ALLOWED:
        problems.append(f"unexpected columns: {extra}")
    if extra and name == "tabular":
        bad = [c for c in extra if not c.startswith(FEATURE_PREFIXES)]
        if bad:
            problems.append(f"tabular feature columns must start with one of {FEATURE_PREFIXES}: {bad}")
    if problems:
        raise ContractError(f"{name}: " + "; ".join(problems))
    return df


def write(df: pl.DataFrame, path: Path | str, name: str) -> Path:
    path = Path(path)
    validate(df, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.write_parquet(path)
    elif path.suffix == ".csv":
        df.write_csv(path)
    else:
        raise ContractError(f"unsupported suffix {path.suffix!r} for {path}")
    return path


def read(path: Path | str, name: str) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, schema_overrides=SCHEMAS[name], try_parse_dates=True)
    else:
        raise ContractError(f"unsupported suffix {path.suffix!r} for {path}")
    return validate(df, name)
