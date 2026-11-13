"""Step 8: evaluate every predictions file: per-run metrics with patient-bootstrap intervals, calibration, F1 at the
development threshold, confusion counts, strata, paired GNN-vs-baseline deltas, LaTeX tables, ROC/PR, reliability
and training-curve figures. Rerunnable: `uv run python scripts/08_evaluate.py [--n-bootstrap N]`.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve, roc_curve

from bridgegnn import config, contract
from bridgegnn.eval import report

ap = argparse.ArgumentParser()
ap.add_argument("--n-bootstrap", type=int, default=None)
args = ap.parse_args()
t0 = time.time()
cfg = config.load()
if args.n_bootstrap:
    cfg["eval"]["n_bootstrap"] = args.n_bootstrap
out = config.outputs_dir(cfg)
ev = out / "eval"
(ev / "tables").mkdir(parents=True, exist_ok=True)
(ev / "figures").mkdir(exist_ok=True)

preds = report.load_predictions(out / "predictions")
rows = contract.read(out / "rows.parquet", "rows")
print(f"{preds.height:,} prediction rows from {preds.select('run_id', 'target', 'seed').unique().height} (run, target, seed) files", flush=True)
tabs = report.summarise(preds, rows, cfg)
for name, df in tabs.items():
    if df.height:
        df.write_csv(ev / "tables" / f"{name}.csv")
agg = tabs["runs_agg"]
agg.with_columns([pl.col(c).round(3) for c in agg.columns if agg[c].dtype == pl.Float64]).write_csv(ev / "tables" / "runs_agg_rounded.csv")
(ev / "tables" / "runs_agg.tex").write_text(report.to_latex(
    agg, ["target", "run_id", "auroc_mean", "auroc_sd", "auroc_seed", "auroc_lo", "auroc_hi", "auprc_mean", "auprc_sd", "brier_mean", "f1_mean", "any_degenerate_threshold", "n_seeds"],
    "Test performance per run: mean and SD over seeds; the interval is the 95\\% patient-level percentile bootstrap of the seed in ci\\_seed (auroc\\_seed is that seed's point estimate).",
    "tab:runs"))
if tabs["paired"].height:
    p0 = tabs["paired"].sort("seed").group_by("run_id", "baseline", "target", maintain_order=True).agg(
        pl.col("seed").first().alias("seed"), pl.col("d_auroc").first(), pl.col("d_auroc_lo").first(), pl.col("d_auroc_hi").first(),
        pl.col("d_auprc").first(), pl.col("d_auprc_lo").first(), pl.col("d_auprc_hi").first(),
        pl.col("d_auroc").mean().alias("d_auroc_mean_seeds"), pl.col("d_auprc").mean().alias("d_auprc_mean_seeds")).sort("target", "run_id")
    (ev / "tables" / "paired.tex").write_text(report.to_latex(
        p0, ["target", "run_id", "baseline", "seed", "d_auroc", "d_auroc_lo", "d_auroc_hi", "d_auprc", "d_auprc_lo", "d_auprc_hi", "d_auroc_mean_seeds"],
        "Paired difference (run minus reference) on identical test patients for the first seed, with its 95\\% patient-level percentile bootstrap interval; the last column is the mean difference over seeds.", "tab:paired"))
else:
    print("note: no pairings computed (no reference run among eval.baselines_for_pairing / eval.paired_runs has predictions yet)")
with pl.Config(tbl_rows=60, tbl_width_chars=170, float_precision=3):
    print(agg.select("target", "run_id", "auroc_mean", "auroc_sd", "auroc_seed", "auroc_lo", "auroc_hi", "auprc_mean", "brier_mean", "f1_mean", "any_degenerate_threshold", "n_seeds"))
    if tabs["paired"].height:
        print(tabs["paired"].group_by("run_id", "baseline", "target").agg(pl.col("d_auroc").mean(), pl.col("d_auprc").mean()).sort("target"))

# ---- figures: ROC / PR for the main runs (seed 0), reliability, training curves ----
main_runs = [r for r in agg["run_id"].unique().to_list() if r.startswith(("G-", "B-"))]
for target in agg["target"].unique().to_list():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for run_id in sorted(main_runs):
        sub = preds.filter((pl.col("run_id") == run_id) & (pl.col("target") == target) & (pl.col("split") == "test"))
        if sub.height == 0:
            continue
        s0 = sub.filter(pl.col("seed") == sub["seed"].min())
        y, p = s0["y"].to_numpy().astype(int), s0["p"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=run_id, lw=1)
        if run_id == "B-prevalence":
            axes[1].axhline(y.mean(), label=run_id, lw=1)
        else:
            axes[1].step(rec, prec, where="post", label=run_id, lw=1)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.6); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].set_title(f"ROC, {target} (first seed)")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title(f"PR, {target} (first seed)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(ev / "figures" / f"roc_pr_{target}.pdf"); plt.close(fig)
    rel = tabs["reliability"]
    if rel.height:
        fig, ax = plt.subplots(figsize=(4, 3.4))
        for run_id in sorted(main_runs):
            r = rel.filter((pl.col("run_id") == run_id) & (pl.col("target") == target))
            if r.height:
                ax.plot(r["p_mean"], r["y_rate"], marker="o", ms=3, lw=1, label=run_id)
        sub = rel.filter((pl.col("target") == target) & pl.col("run_id").is_in(main_runs))
        lim = max(0.05, float(max(sub["p_mean"].max() or 0, sub["y_rate"].max() or 0)) * 1.1)
        ax.plot([0, lim], [0, lim], "k--", lw=0.6); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel("mean predicted p (10 quantile bins)"); ax.set_ylabel("observed rate"); ax.set_title(f"reliability, {target} (first seed)")
        ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(ev / "figures" / f"reliability_{target}.pdf"); plt.close(fig)
for f in sorted((out / "logs").glob("06_G-*_seed0.json")):
    c = json.loads(f.read_text())
    sel = c.get("selection", [])
    if not sel:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for k, fold in enumerate(sel):
        axes[0].plot(fold["train_loss"], lw=0.8, label=f"fold {k} train"); axes[0].plot(fold.get("val_loss", []), lw=0.8, ls="--", label=f"fold {k} val")
        axes[1].plot([v if v is not None else np.nan for v in fold["val_auprc"]], lw=0.8)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation AUPRC")
    axes[0].set_title(f.stem.replace("06_", "").replace("_seed0", "") + " (selection folds)"); axes[0].legend(fontsize=5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(ev / "figures" / f"curves_{f.stem.replace('06_', '').replace('_seed0', '')}.pdf"); plt.close(fig)
print(f"wrote {ev}/tables and {ev}/figures  ({time.time()-t0:.0f}s)")
