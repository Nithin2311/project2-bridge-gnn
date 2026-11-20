"""Compare two artifact trees (a reference run and a fresh-clone re-run) table by table.

Status per (file, column): exact | within_tol | exceeds | missing_file | missing_column. Numeric differences are
tolerated only in the torch-dependent tables, for rows whose model is not a deterministic tabular baseline, and only
up to `tol`. String columns (incl. the selected hyperparameters in params_json) must match exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

CSV_DIRS = ["results", "eval/tables", "layers", "motifs", "pre2020", "cohort"]
NPY_FILES = ["node_x.npy"]
TORCH_FILES = {"results/gnn.csv", "eval/tables/runs.csv", "eval/tables/runs_agg.csv", "eval/tables/runs_agg_rounded.csv",
               "eval/tables/paired.csv", "eval/tables/reliability.csv", "eval/tables/strata.csv"}
DETERMINISTIC_MODELS = {"prevalence", "lr", "hgb", "hgb_all", "bridge_lr", "cov_lr"}
STATUSES = ["exact", "within_tol", "exceeds", "missing_file", "missing_column"]
COLUMNS = ["file", "column", "status", "max_abs_diff", "n_rows_old", "n_rows_new", "note"]


def _row(file, column, status, diff=None, n_old=None, n_new=None, note=""):
    return {"file": file, "column": column, "status": status, "max_abs_diff": diff,
            "n_rows_old": n_old, "n_rows_new": n_new, "note": note}


def _is_numeric(dtype) -> bool:
    return dtype.is_numeric()


def _deterministic_mask(df: pl.DataFrame) -> np.ndarray:
    """True for rows whose numbers must match exactly (tabular baselines); torch rows may drift within tol."""
    if "model" in df.columns:
        return df["model"].is_in(list(DETERMINISTIC_MODELS)).fill_null(False).to_numpy()
    if "run_id" in df.columns:
        return df["run_id"].str.starts_with("B-").to_numpy() | df["run_id"].str.ends_with("-hgb").to_numpy()
    return np.ones(df.height, dtype=bool)


def _drop_runs(df: pl.DataFrame, prefixes: list[str]) -> pl.DataFrame:
    if not prefixes or "run_id" not in df.columns:
        return df
    keep = pl.lit(True)
    for p in prefixes:
        keep = keep & ~pl.col("run_id").cast(pl.Utf8).str.starts_with(p)
    return df.filter(keep)


def _compare_csv(rel: str, old_f: Path, new_f: Path, tol: float, exclude_runs: list[str] | None = None) -> list[dict]:
    a, b = pl.read_csv(old_f, infer_schema_length=None), pl.read_csv(new_f, infer_schema_length=None)
    a, b = _drop_runs(a, exclude_runs or []), _drop_runs(b, exclude_runs or [])
    rows = []
    for c in a.columns:
        if c not in b.columns:
            rows.append(_row(rel, c, "missing_column", n_old=a.height, n_new=b.height, note="absent in new"))
    for c in b.columns:
        if c not in a.columns:
            rows.append(_row(rel, c, "missing_column", n_old=a.height, n_new=b.height, note="absent in old"))
    common = [c for c in a.columns if c in b.columns]
    keys = [c for c in common if not _is_numeric(a[c].dtype) and c != "params_json"]
    # seed and other integer identifiers are keys too when present
    for c in ("seed", "n_seeds", "ci_seed"):
        if c in common and c not in keys:
            keys.append(c)
    if not keys:
        keys = []
    a2, b2 = _keys_as_text(a.select(common), keys), _keys_as_text(b.select(common), keys)
    if keys and (a2.select(keys).unique().height != a2.height or b2.select(keys).unique().height != b2.height):
        keys = []   # keys do not identify rows; fall back to positional comparison
    if keys:
        joined = a2.join(b2, on=keys, how="inner", suffix="_new")
        if joined.height != a.height or joined.height != b.height:
            rows.append(_row(rel, "*", "exceeds", n_old=a.height, n_new=b.height,
                             note=f"rows differ: {a.height} old, {b.height} new, {joined.height} matched on {keys}"))
    else:
        if a.height != b.height:
            rows.append(_row(rel, "*", "exceeds", n_old=a.height, n_new=b.height, note="rows differ: row counts (positional)"))
            return rows
        joined = pl.concat([a2, b2.rename({c: f"{c}_new" for c in common})], how="horizontal")
    det = _deterministic_mask(joined)
    tolerant_file = rel in TORCH_FILES
    for c in common:
        if c in keys:
            continue
        x, y = joined[c], joined[f"{c}_new"]
        if _is_numeric(x.dtype) and _is_numeric(y.dtype):
            xv, yv = x.cast(pl.Float64).to_numpy(), y.cast(pl.Float64).to_numpy()
            with np.errstate(invalid="ignore"):
                d = np.abs(xv - yv)
            d = np.where((xv == yv) | (np.isnan(xv) & np.isnan(yv)), 0.0, d)   # equal cells (incl. inf, null) are exact
            d = np.where(np.isnan(d), np.inf, d)                               # null vs value, or inf vs finite
            diff = float(d.max()) if d.size else 0.0
            if diff == 0.0:
                status = "exact"
            elif tolerant_file and diff <= tol and not (d[det] > 0).any():
                status = "within_tol"
            else:
                status = "exceeds"
            rows.append(_row(rel, c, status, diff, a.height, b.height))
        else:
            xs, ys = x.cast(pl.Utf8).fill_null(""), y.cast(pl.Utf8).fill_null("")
            if c == "params_json":
                xs = xs.map_elements(_canon_json, return_dtype=pl.Utf8)
                ys = ys.map_elements(_canon_json, return_dtype=pl.Utf8)
            n_diff = int((xs != ys).sum())
            rows.append(_row(rel, c, "exact" if n_diff == 0 else "exceeds", float(n_diff), a.height, b.height,
                             "" if n_diff == 0 else f"{n_diff} cells differ"))
    return rows


def _keys_as_text(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Key columns as text with nulls made explicit, so a null key still aligns with the same null in the other tree."""
    return df.with_columns([pl.col(k).cast(pl.Utf8).fill_null("<null>") for k in keys])


# machine-specific paths and row identity inside params_json carry no hyperparameter information
IGNORED_PARAM_KEYS = {"node_features_dir", "layer_file", "subsample_file", "run_id", "target"}


def _canon_json(s: str) -> str:
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        return s
    if isinstance(d, dict):
        # null-valued keys are flags the CLI gained later and did not set: the same configuration as an absent key
        d = {k: v for k, v in d.items() if k not in IGNORED_PARAM_KEYS and v is not None}
    return json.dumps(d, sort_keys=True)


def _compare_npy(rel: str, old_f: Path, new_f: Path) -> dict:
    a, b = np.load(old_f), np.load(new_f)
    if a.shape != b.shape:
        return _row(rel, "matrix", "exceeds", None, a.shape[0], b.shape[0], f"shape {a.shape} vs {b.shape}")
    diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
    return _row(rel, "matrix", "exact" if diff == 0.0 else "exceeds", diff, a.shape[0], b.shape[0])


def compare_dirs(old: Path, new: Path, tol: float = 0.02, exclude_runs: list[str] | None = None) -> pl.DataFrame:
    """`exclude_runs`: run_id prefixes dropped from both trees before comparing (e.g. runs whose inputs changed by design)."""
    old, new = Path(old), Path(new)
    rows: list[dict] = []
    rels = set()
    for d in CSV_DIRS:
        for root in (old, new):
            rels.update(str(p.relative_to(root)) for p in (root / d).glob("*.csv"))
    for rel in sorted(rels):
        o, n = old / rel, new / rel
        if not o.exists():
            rows.append(_row(rel, "*", "missing_file", note="absent in old"))
        elif not n.exists():
            rows.append(_row(rel, "*", "missing_file", note="absent in new"))
        else:
            rows.extend(_compare_csv(rel, o, n, tol, exclude_runs))
    for rel in NPY_FILES:
        o, n = old / rel, new / rel
        if o.exists() and n.exists():
            rows.append(_compare_npy(rel, o, n))
        elif o.exists() or n.exists():
            rows.append(_row(rel, "*", "missing_file", note="absent in " + ("new" if o.exists() else "old")))
    return pl.DataFrame(rows, schema={"file": pl.Utf8, "column": pl.Utf8, "status": pl.Utf8, "max_abs_diff": pl.Float64,
                                      "n_rows_old": pl.Int64, "n_rows_new": pl.Int64, "note": pl.Utf8})


def summary(df: pl.DataFrame) -> dict:
    counts = {s: 0 for s in STATUSES}
    for s, n in df.group_by("status").len().iter_rows():
        counts[s] = int(n)
    return counts


def to_markdown(df: pl.DataFrame, only_nonexact: bool = True) -> str:
    s = summary(df)
    head = "Summary: " + ", ".join(f"{k} {v}" for k, v in s.items()) + f" (of {df.height} comparisons)\n\n"
    sub = df.filter(pl.col("status") != "exact") if only_nonexact else df
    lines = ["| file | column | status | max abs diff | rows old | rows new | note |", "|---|---|---|---|---|---|---|"]
    for r in sub.iter_rows(named=True):
        diff = "" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.4g}"
        lines.append(f"| {r['file']} | {r['column']} | {r['status']} | {diff} | {r['n_rows_old'] or ''} | {r['n_rows_new'] or ''} | {r['note']} |")
    if sub.height == 0:
        lines.append("| (every comparison exact) | | | | | | |")
    return head + "\n".join(lines) + "\n"
