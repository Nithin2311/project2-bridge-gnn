"""Step 08 table builders: from predictions files (+ rows for strata) to the rubric's tables."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

from .. import contract
from . import metrics

STRATA = ["sex", "age_band", "size_bin", "index_year"]


def load_predictions(d: Path) -> pl.DataFrame:
    files = sorted(Path(d).glob("*.parquet"))
    if not files:
        return pl.DataFrame({c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["predictions"].items()})
    return pl.concat([contract.read(f, "predictions") for f in files])


def dev_threshold(one_run: pl.DataFrame) -> float:
    """F1-maximising threshold from the development out-of-fold rows of one (run_id, target, seed)."""
    dev = one_run.filter(pl.col("split") != "test")
    return metrics.f1_threshold(dev["y"].to_numpy().astype(int), dev["p"].to_numpy())


def _with_strata(preds: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    r = rows.select("person_id", "index_date", "target", "sex", "age_at_index", "n_codes_lb1095", "index_year").unique(
        subset=["person_id", "index_date", "target"])
    j = preds.join(r, on=["person_id", "index_date", "target"], how="left")
    missing = j.filter(pl.col("age_at_index").is_null()).height
    if missing:
        raise ValueError(f"{missing} prediction rows found no matching row in rows.parquet (person_id, index_date, target)")
    return j.with_columns(
        pl.when(pl.col("age_at_index") >= 65).then(pl.lit("65+")).otherwise(pl.lit("<65")).alias("age_band"),
        pl.when(pl.col("n_codes_lb1095") < 3).then(pl.lit("<3")).when(pl.col("n_codes_lb1095") < 10).then(pl.lit("3-9"))
        .otherwise(pl.lit("10+")).alias("size_bin"),
        pl.col("index_year").cast(pl.Utf8))


def _pairings(cfg_eval: dict, run_ids: set[str]) -> list[tuple[str, str]]:
    """(run, reference) pairs: every non-baseline run vs each configured baseline, plus explicit paired_runs."""
    baselines = [b for b in cfg_eval.get("baselines_for_pairing", []) if b in run_ids]
    pairs = [(r, b) for r in sorted(run_ids) for b in baselines if r != b and r not in baselines]
    for a, b in cfg_eval.get("paired_runs", []) or []:
        if a in run_ids and b in run_ids and (a, b) not in pairs:
            pairs.append((a, b))
    return pairs


def summarise(preds: pl.DataFrame, rows: pl.DataFrame, cfg: dict) -> dict[str, pl.DataFrame]:
    E = cfg["eval"]
    nb, seed, min_pos, bins = int(E["n_bootstrap"]), int(E["seed"]), int(E["min_stratum_pos"]), int(E["calib_bins"])
    P = _with_strata(preds, rows)
    runs, rel, strata, paired = [], [], [], []
    keys = sorted({(r, t, s) for r, t, s in P.select("run_id", "target", "seed").unique().iter_rows()})
    ci_seed = {(r, t): min(k[2] for k in keys if k[0] == r and k[1] == t) for r, t, _ in keys}
    for run_id, target, s in keys:
        one = P.filter((pl.col("run_id") == run_id) & (pl.col("target") == target) & (pl.col("seed") == s))
        test = one.filter(pl.col("split") == "test")
        y, p, pid = test["y"].to_numpy().astype(int), test["p"].to_numpy(), test["person_id"].to_numpy()
        try:
            thr, src = dev_threshold(one), "dev_oof"
        except ValueError:
            thr, src = 0.5, "fallback"
        cal = metrics.calibration(y, p)
        conf = metrics.confusion(y, p, thr)
        n_pred_pos = conf["tp"] + conf["fp"]
        degenerate = src == "fallback" or n_pred_pos == 0 or n_pred_pos == len(y)
        row = {"run_id": run_id, "target": target, "seed": int(s), "model": one["model"][0],
               "auroc": metrics.auroc(y, p), "auprc": metrics.auprc(y, p), **cal, **conf,
               "threshold_source": src, "pred_pos_rate": n_pred_pos / max(len(y), 1), "degenerate_threshold": bool(degenerate),
               "n": len(y), "pos": int(y.sum())}
        if src == "fallback":
            row["f1"] = None
            row["precision"] = None
            row["recall"] = None
        if s == ci_seed[(run_id, target)]:
            _, row["auroc_lo"], row["auroc_hi"] = metrics.bootstrap_ci(y, p, pid, metrics.auroc, nb, seed)
            _, row["auprc_lo"], row["auprc_hi"] = metrics.bootstrap_ci(y, p, pid, metrics.auprc, nb, seed)
            rel.append(metrics.reliability(y, p, bins).with_columns(pl.lit(run_id).alias("run_id"), pl.lit(target).alias("target"), pl.lit(int(s)).alias("seed")))
            strata.append(metrics.stratified(test, STRATA, min_pos).with_columns(pl.lit(run_id).alias("run_id"), pl.lit(target).alias("target"), pl.lit(int(s)).alias("seed")))
        else:
            row["auroc_lo"] = row["auroc_hi"] = row["auprc_lo"] = row["auprc_hi"] = None
        runs.append(row)
    runs_df = pl.DataFrame(runs)
    first = runs_df.filter(pl.col("auroc_lo").is_not_null()).select(
        "run_id", "target", pl.col("seed").alias("ci_seed"), pl.col("auroc").alias("auroc_seed"), pl.col("auprc").alias("auprc_seed"),
        "auroc_lo", "auroc_hi", "auprc_lo", "auprc_hi")
    agg = (runs_df.group_by("run_id", "target", "model").agg(
        pl.col("auroc").mean().alias("auroc_mean"), pl.col("auroc").std().alias("auroc_sd"),
        pl.col("auprc").mean().alias("auprc_mean"), pl.col("auprc").std().alias("auprc_sd"),
        pl.col("brier").mean().alias("brier_mean"), pl.col("brier").std().alias("brier_sd"),
        pl.col("slope").mean().alias("slope_mean"), pl.col("citl").mean().alias("citl_mean"),
        pl.col("f1").mean().alias("f1_mean"), pl.col("f1").std().alias("f1_sd"),
        pl.col("degenerate_threshold").any().alias("any_degenerate_threshold"),
        pl.len().alias("n_seeds"), pl.col("n").first(), pl.col("pos").first())
           .join(first, on=["run_id", "target"], how="left")
           .with_columns(pl.col("ci_seed").alias("auroc_ci_seed"))
           .sort("target", "run_id"))
    for run_id, ref in _pairings(E, set(P["run_id"].to_list())):
        for target, s in sorted({(t, s) for r, t, s in keys if r == run_id}):
            a = P.filter((pl.col("run_id") == run_id) & (pl.col("target") == target) & (pl.col("seed") == s) & (pl.col("split") == "test"))
            b = P.filter((pl.col("run_id") == ref) & (pl.col("target") == target) & (pl.col("seed") == s) & (pl.col("split") == "test"))
            if a.height == 0 or b.height == 0:
                continue
            m = a.join(b.select("person_id", "index_date", pl.col("p").alias("p_b"), pl.col("y").alias("y_b")), on=["person_id", "index_date"], how="inner")
            if m.height == 0:
                continue
            if (m["y"] != m["y_b"]).any():
                raise ValueError(f"labels differ between {run_id} and {ref} on shared rows ({target}, seed {s})")
            d = metrics.paired_delta(m["y"].to_numpy().astype(int), m["p"].to_numpy(), m["p_b"].to_numpy(), m["person_id"].to_numpy(), nb, seed)
            paired.append({"run_id": run_id, "baseline": ref, "target": target, "seed": int(s), "n": m.height, **d})
    schema = {"run_id": pl.Utf8, "baseline": pl.Utf8, "target": pl.Utf8, "seed": pl.Int64, "n": pl.Int64,
              "d_auroc": pl.Float64, "d_auroc_lo": pl.Float64, "d_auroc_hi": pl.Float64, "d_auprc": pl.Float64, "d_auprc_lo": pl.Float64, "d_auprc_hi": pl.Float64}
    paired_df = pl.DataFrame(paired, schema=schema) if paired else pl.DataFrame(schema=schema)
    return {"runs": runs_df, "runs_agg": agg,
            "reliability": pl.concat(rel) if rel else pl.DataFrame(),
            "strata": pl.concat(strata) if strata else pl.DataFrame(),
            "paired": paired_df}


def _esc(s: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", s)


def to_latex(df: pl.DataFrame, cols: list[str], caption: str, label: str, decimals: int = 3) -> str:
    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return "n/a" if np.isnan(v) else f"{v:.{decimals}f}"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return _esc(str(v))
    head = " & ".join(_esc(c) for c in cols) + r" \\"
    body = "\n".join(" & ".join(fmt(r[c]) for c in cols) + r" \\" for r in df.select(cols).iter_rows(named=True))
    return ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n"
            "\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
