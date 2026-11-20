"""Report figure: ROC, precision-recall and reliability for a curated set of models on one target (first seed).

The all-model figures written by step 08 stay in the artifacts; this one is what the report shows, with five lines at
most, distinct colours and dashes, and panels large enough to read in a two-column layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import precision_recall_curve, roc_curve

from . import metrics

LABELS = {"G-gcn": "GCN", "G-sage": "GraphSAGE", "G-gat": "GAT", "B2": "Bipartite (hetero)", "B-prevalence": "Prevalence",
          "B-lr": "Code-bag LR", "B-hgb": "Graph-code boosting", "B-hgb_all": "All-code boosting", "B-mlp": "MLP",
          "B-mlp_pool": "Pooled MLP", "B-bridge_lr": "Bridge LR", "B-cov_lr": "Covariates-only LR"}
# curated set for the report: the test-best GNN, the ablation backbone, the two strongest flat comparators, the reference
REPORT_RUNS = ["G-gat", "G-sage", "B-bridge_lr", "B-cov_lr", "B-hgb_all", "B-prevalence"]
STYLE = {"G-gat": ("#b2182b", "-"), "G-sage": ("#ef8a62", "-"), "G-gcn": ("#d6604d", "-."), "B-cov_lr": ("#2166ac", "--"),
         "B-bridge_lr": ("#4393c3", "--"), "B-hgb_all": ("#1b7837", ":"), "B-hgb": ("#5aae61", ":"), "B-mlp_pool": ("#762a83", "-."),
         "B-prevalence": ("#4d4d4d", (0, (1, 1)))}
TARGET_NAMES = {"cardiometabolic": "cardiometabolic core", "cancer_hematologic": "hematology-associated core",
                "mental_health": "mental-health and sleep core", "cancer_metastatic": "metastatic-associated core"}


def main_figure(preds: pl.DataFrame, target: str, runs: list[str], out: Path, bins: int = 10) -> list[str]:
    """Draw the three-panel figure for `runs` that have first-seed test predictions on `target`; returns the runs drawn."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    drawn = []
    for run_id in runs:
        sub = preds.filter((pl.col("run_id") == run_id) & (pl.col("target") == target) & (pl.col("split") == "test"))
        if sub.height == 0:
            continue
        s0 = sub.filter(pl.col("seed") == sub["seed"].min())
        y, p = s0["y"].to_numpy().astype(int), s0["p"].to_numpy()
        colour, dash = STYLE.get(run_id, ("black", "-"))
        label = LABELS.get(run_id, run_id)
        lw = 2.0 if run_id.startswith("G-") else 1.6
        if run_id == "B-prevalence":
            axes[0].plot([0, 1], [0, 1], color=colour, ls=dash, lw=1.2, label=label)
            axes[1].axhline(y.mean(), color=colour, ls=dash, lw=1.2, label=label)
        else:
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            axes[0].plot(fpr, tpr, color=colour, ls=dash, lw=lw, label=f"{label} ({metrics.auroc(y, p):.2f})")
            axes[1].step(rec, prec, where="post", color=colour, ls=dash, lw=lw, label=f"{label} ({metrics.auprc(y, p):.2f})")
            rel = metrics.reliability(y, p, bins)
            axes[2].plot(rel["p_mean"], rel["y_rate"], color=colour, ls=dash, lw=lw, marker="o", ms=3.5, label=label)
        drawn.append(run_id)
    name = TARGET_NAMES.get(target, target)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].set_title("ROC (AUROC in legend)")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("precision-recall (AUPRC in legend)")
    # the precision spike at near-zero recall (a handful of rows) would otherwise set the scale; show the curves instead
    top = 0.05
    for line in axes[1].get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs) > 1:
            body = [yv for xv, yv in zip(xs, ys) if xv >= 0.05]
            if body:
                top = max(top, max(body))
    axes[1].set_ylim(0, min(1.0, top * 1.15))
    lim = 0.05
    for line in axes[2].get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs):
            lim = max(lim, float(max(xs)), float(max(ys)))
    lim *= 1.08
    axes[2].plot([0, lim], [0, lim], color="#4d4d4d", ls=(0, (1, 1)), lw=1.0, label="perfect calibration")
    axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim); axes[2].set_aspect("equal")
    axes[2].set_xlabel(f"mean predicted probability ({bins} quantile bins)"); axes[2].set_ylabel("observed rate"); axes[2].set_title("reliability")
    for ax, loc in zip(axes, ("lower right", "upper right", "upper left")):
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.legend(frameon=False, loc=loc)
    fig.suptitle(f"{name}, test rows, first seed", y=1.02, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return drawn
