"""Generate the LaTeX result fragments for the baseline, C1 and D1 sections from step 08's tables.

Inputs (defaults = the committed aggregates): outputs/eval/tables/runs_agg.csv (every run: G-*, B-*, C1-*, D1-*),
outputs/eval/tables/paired.csv (step 08 paired deltas on identical test rows), outputs/results/baselines.csv
(selected configurations). Run after `scripts/08_evaluate.py` with both the GNN and baseline predictions present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from bridgegnn import config, contract

TARGETS = {"cardiometabolic": "CM", "cancer_hematologic": "Heme", "mental_health": "MH"}
BASELINES = {
    "B-prevalence": "Prevalence", "B-lr": "Logistic regression", "B-hgb": "Graph-code boosting",
    "B-hgb_all": "All-code boosting", "B-mlp": "MLP", "B-mlp_pool": "Pooled MLP",
    "B-bridge_lr": "Bridge logistic regression", "B-cov_lr": "Covariates-only logistic regression",
}
DETERMINISTIC = {"B-prevalence", "B-lr", "B-hgb", "B-hgb_all", "B-bridge_lr", "B-cov_lr"}
GNNS = {"G-gcn": "GCN", "G-sage": "GraphSAGE", "G-gat": "GAT"}
HETERO = {"B2": "Bipartite (hetero)"}   # extra-credit variant: listed with the GNNs when its five-seed rows exist
REWIRES = [f"D1-rewire{i}" for i in range(5)]
SCALE = {
    "C1-25": r"GraphSAGE, 25\%", "C1-50": r"GraphSAGE, 50\%", "G-sage": r"GraphSAGE, 100\%",
    "C1-25-hgb": r"Boosting, 25\%", "C1-50-hgb": r"Boosting, 50\%", "B-hgb": r"Boosting, 100\%",
}
PAIR_REFS = ["B-hgb_all", "B-mlp_pool"]


def tex_row(label: str, fields: list[str]) -> str:
    return " & ".join([label, *fields]) + r" \\"


def table_map(file: Path) -> dict:
    return {(r["run_id"], r["target"]): r for r in pl.read_csv(file).iter_rows(named=True)}


def format_params(row: dict) -> str:
    p = json.loads(row["params_json"])
    model = row["model"]
    if model == "prevalence":
        return "Development prior"
    if model in {"lr", "bridge_lr", "cov_lr"}:
        return rf"$C={p['C']:g}$"
    if model in {"hgb", "hgb_all"}:
        return rf"$\eta={p['learning_rate']:g}$, {p['max_leaf_nodes']} leaves, {p['max_iter']} rounds"
    widths = "--".join(map(str, p["hidden_layer_sizes"]))
    alpha = r"10^{-4}" if p["alpha"] == 1e-4 else r"10^{-3}"
    return rf"{widths} units, $\alpha={alpha}$, {p['refit_epochs']} epochs"


def main() -> None:
    root = config.REPO_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=root / "outputs/eval/tables/runs_agg.csv")
    ap.add_argument("--paired", type=Path, default=root / "outputs/eval/tables/paired.csv")
    ap.add_argument("--baseline-results", type=Path, default=root / "outputs/results/baselines.csv")
    ap.add_argument("--out", type=Path, default=root / "report")
    args = ap.parse_args()
    stats = table_map(args.runs)
    rows = contract.read(args.baseline_results, "results")
    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, content):
        (args.out / f"{name}.tex").write_text(content + "\n", encoding="utf-8")

    def stat(run, target, key):
        return stats[(run, target)][key]

    def msd(run, target, metric):
        r = stats[(run, target)]
        return f"{r[f'{metric}_mean']:.3f}" + r"\(\pm\)" + f"{r[f'{metric}_sd'] or 0.0:.3f}"

    def ci(run, target, metric):
        r = stats[(run, target)]
        return f"{r[f'{metric}_seed']:.3f} [{r[f'{metric}_lo']:.3f}, {r[f'{metric}_hi']:.3f}]"

    for run in {*BASELINES, *SCALE, *REWIRES, "D1-shuffle", *GNNS}:
        for target in TARGETS:
            row = stats.get((run, target))
            if row is None or row["n_seeds"] != 5:
                raise ValueError(f"missing five-seed aggregate for {run} / {target}")
    hetero = {r: n for r, n in HETERO.items() if all(stats.get((r, t), {}).get("n_seeds") == 5 for t in TARGETS)}
    if not hetero:
        print("note: no five-seed B2 (bipartite hetero) rows; fragments list the three homogeneous GNNs only", file=sys.stderr)
    graph_models = {**GNNS, **hetero}

    def label(run, name):
        return name + (r"$^{\dagger}$" if run in DETERMINISTIC else "")

    # main table: AUROC and AUPRC, mean and SD over seeds, baselines then GNNs
    main_rows = [tex_row(label(run, name), [f"{msd(run, t, 'auroc')} & {msd(run, t, 'auprc')}" for t in TARGETS])
                 for run, name in {**BASELINES, **graph_models}.items()]
    write("baseline-rows", "\n".join(main_rows))
    # appendix table: seed-0 point estimate with the 95% patient-bootstrap interval
    ci_rows = [tex_row(label(run, name), [f"{ci(run, t, 'auroc')} & {ci(run, t, 'auprc')}" for t in TARGETS])
               for run, name in {**BASELINES, **graph_models}.items()]
    write("baseline-ci-rows", "\n".join(ci_rows))
    write("scale-ap-rows", "\n".join(tex_row(lab, [f"{msd(run, t, 'auroc')} & {msd(run, t, 'auprc')}" for t in TARGETS])
                                     for run, lab in SCALE.items()))
    ranges = {t: (min(stat(r, t, "auprc_mean") for r in REWIRES), max(stat(r, t, "auprc_mean") for r in REWIRES)) for t in TARGETS}
    ranges_roc = {t: (min(stat(r, t, "auroc_mean") for r in REWIRES), max(stat(r, t, "auroc_mean") for r in REWIRES)) for t in TARGETS}
    write("null-ap-rows", "\n".join([
        tex_row("Layer U", [f"{msd('G-sage', t, 'auroc')} & {msd('G-sage', t, 'auprc')}" for t in TARGETS]),
        tex_row("Degree-preserving rewires (range of means)", [f"{ranges_roc[t][0]:.3f}--{ranges_roc[t][1]:.3f} & {ranges[t][0]:.3f}--{ranges[t][1]:.3f}" for t in TARGETS]),
        tex_row("Edge shuffle", [f"{msd('D1-shuffle', t, 'auroc')} & {msd('D1-shuffle', t, 'auprc')}" for t in TARGETS]),
    ]))

    notes = []
    for t, short in TARGETS.items():
        best_b = max(BASELINES, key=lambda r: stat(r, t, "auroc_mean"))
        best_g = max(GNNS, key=lambda r: stat(r, t, "auroc_mean"))
        notes.append(f"For {short}, the strongest baseline by mean AUROC was {BASELINES[best_b].lower()} "
                     f"({stat(best_b, t, 'auroc_mean'):.3f}, AUPRC {stat(best_b, t, 'auprc_mean'):.3f}) and the strongest GNN was "
                     f"{GNNS[best_g]} ({stat(best_g, t, 'auroc_mean'):.3f}, AUPRC {stat(best_g, t, 'auprc_mean'):.3f}).")
    def overlaps(a, b, t):
        return stat(a, t, "auroc_lo") <= stat(b, t, "auroc_hi") and stat(b, t, "auroc_lo") <= stat(a, t, "auroc_hi")
    non_overlap = [f"{short} vs {graph_models[g]}" for t, short in TARGETS.items() for g in graph_models if not overlaps("B-bridge_lr", g, t)]
    notes.append("Bridge logistic regression includes utilization and demographics, so its discrimination combines their signal "
                 "with the three bridge summaries; its first-seed AUROC interval " +
                 ("overlaps every graph model's on every target" if not non_overlap else
                  "overlaps every graph model's except " + ", ".join(non_overlap)) + " (Table~\\ref{tab:baseline-ci}).")
    for r, n in hetero.items():
        notes.append(f"The bipartite heterogeneous variant ({r}, listed as {n}) reached mean AUROC " +
                     ", ".join(f"{stat(r, t, 'auroc_mean'):.3f} ({short})" for t, short in TARGETS.items()) + ".")
    if args.paired.exists():
        paired = pl.read_csv(args.paired).filter((pl.col("run_id") == "G-sage") & pl.col("baseline").is_in(PAIR_REFS) & (pl.col("seed") >= 0))
        if paired.height:
            s0 = paired.filter(pl.col("seed") == paired["seed"].min())
            excl = []
            for t, short in TARGETS.items():
                parts = []
                for ref in PAIR_REFS:
                    r = s0.filter((pl.col("target") == t) & (pl.col("baseline") == ref))
                    if r.height == 0:
                        continue
                    r = r.row(0, named=True)
                    parts.append(rf"{BASELINES[ref].lower()}: AUROC {r['d_auroc']:+.3f} [{r['d_auroc_lo']:+.3f}, {r['d_auroc_hi']:+.3f}], "
                                 rf"AUPRC {r['d_auprc']:+.3f} [{r['d_auprc_lo']:+.3f}, {r['d_auprc_hi']:+.3f}]")
                    if r["d_auroc_lo"] > 0 or r["d_auroc_hi"] < 0:
                        excl.append(f"{short} AUROC vs {BASELINES[ref].lower()}")
                    if r["d_auprc_lo"] > 0 or r["d_auprc_hi"] < 0:
                        excl.append(f"{short} AUPRC vs {BASELINES[ref].lower()}")
                if parts:
                    notes.append(f"Paired GraphSAGE-minus-baseline differences on identical test patients (first seed, {short}, "
                                 f"95\\% patient-bootstrap intervals): " + "; ".join(parts) + ".")
            notes.append("First-seed intervals excluding zero: " + (", ".join(excl) if excl else "none") + "; the seed-mean intervals are in the supplement.")
    write("baseline-results-summary", " ".join(notes))

    scale_notes = []
    for lab, full, quarter in [("GraphSAGE", "G-sage", "C1-25"), ("boosting", "B-hgb", "C1-25-hgb")]:
        diffs = ", ".join(f"{stat(full, t, 'auroc_mean') - stat(quarter, t, 'auroc_mean'):+.3f}" for t in TARGETS)
        scale_notes.append(f"The full-minus-quarter AUROC differences for {lab} were {diffs} (CM, Heme, MH).")
    write("scale-summary", " ".join(scale_notes) + " Both families were trained on the same seeded patient subsets.")

    null_notes = []
    for t, short in TARGETS.items():
        lo, hi = ranges_roc[t]
        real, shuffled = stat("G-sage", t, "auroc_mean"), stat("D1-shuffle", t, "auroc_mean")
        rel = "above" if real > hi else "below" if real < lo else "within"
        null_notes.append(f"For {short}, the observed-edge mean AUROC ({real:.3f}) was {rel} the rewire range ({lo:.3f}--{hi:.3f}); "
                          f"the shuffled-edge mean was {shuffled:.3f}.")
    inside = all(lo <= stat("G-sage", t, "auroc_mean") <= hi for t, (lo, hi) in ranges_roc.items())
    write("null-summary", " ".join(null_notes) + (" The observed graph stays within the rewire range on every target, so the "
          "message-passing edges add no measurable discrimination beyond the node features." if inside else
          " These contrasts are descriptive because the experiment used five rewired graphs."))

    # supplement: calibration and thresholded metrics per model and target (Brier skill against the prevalence comparator)
    def skill(run, t):
        ref = stat("B-prevalence", t, "brier_mean")
        return 1.0 - stat(run, t, "brier_mean") / ref if ref else float("nan")
    metric_rows = []
    for run, name in {**BASELINES, **graph_models}.items():
        cells = []
        for t in TARGETS:
            slope = "n/a" if run == "B-prevalence" else f"{stat(run, t, 'slope_mean'):.2f}"
            f1 = stats[(run, t)]["f1_mean"]
            cells.append(f"{stat(run, t, 'brier_mean'):.3f} & {100 * skill(run, t):+.1f}\\% & {slope} & {stat(run, t, 'citl_mean'):+.2f} & "
                         + ("n/a" if f1 is None else f"{f1:.2f}"))
        metric_rows.append(tex_row(label(run, name), cells))
    write("metrics-rows", "\n".join(metric_rows))

    # supplement: headline paired differences, seed-pooled interval and the number of per-seed intervals excluding zero
    if args.paired.exists():
        pr = pl.read_csv(args.paired)
        pooled = pr.filter(pl.col("seed") == -1)
        per = pr.filter(pl.col("seed") >= 0).group_by("run_id", "baseline", "target").agg(
            pl.len().alias("n_seeds"), ((pl.col("d_auroc_lo") > 0) | (pl.col("d_auroc_hi") < 0)).sum().alias("n_excl"))
        pooled = pooled.join(per, on=["run_id", "baseline", "target"], how="left")
        names = {**BASELINES, **GNNS, **HETERO, "D1-shuffle": "Edge shuffle", "D1-rewire0": "Rewiring 0", "D1-empty": "No edges",
                 "A1a": "No BRIDGE features", "A1b": "No BRIDGE, codes masked", "B1": "Directed layer", "B5-U": "Layer U (rows from 2020)",
                 "B5-P": "Pre-2020 graph", "BT-U": "Backtest, Layer U", "BT-P": "Backtest, pre-2020 graph"}
        order = [("G-sage", "B-bridge_lr"), ("G-sage", "B-hgb"), ("G-sage", "B-hgb_all"), ("G-sage", "B-mlp_pool"), ("G-sage", "B-cov_lr"),
                 ("G-gcn", "B-bridge_lr"), ("G-gat", "B-bridge_lr"), ("B2", "G-sage"), ("B2", "B-bridge_lr"),
                 ("A1a", "G-sage"), ("A1b", "G-sage"), ("B1", "G-sage"), ("D1-shuffle", "G-sage"), ("D1-rewire0", "G-sage"), ("D1-empty", "G-sage"),
                 ("B-bridge_lr", "B-cov_lr"), ("B-bridge_lr", "B-hgb_all"), ("B5-P", "B5-U"), ("BT-P", "BT-U")]
        prow = []
        for run, ref in order:
            cells = []
            for t in TARGETS:
                r = pooled.filter((pl.col("run_id") == run) & (pl.col("baseline") == ref) & (pl.col("target") == t))
                if r.height == 0:
                    cells.append("-- & --")
                    continue
                r = r.row(0, named=True)
                cells.append(f"{r['d_auroc']:+.3f} [{r['d_auroc_lo']:+.3f}, {r['d_auroc_hi']:+.3f}] & {int(r['n_excl'] or 0)}/{int(r['n_seeds'] or 0)}")
            if any(c != "-- & --" for c in cells):
                prow.append(tex_row(f"{names.get(run, run)} $-$ {names.get(ref, ref)}", cells))
        write("paired-seeds-rows", "\n".join(prow))

    configs = []
    for run, name in BASELINES.items():
        for t, short in TARGETS.items():
            sub = rows.filter((pl.col("run_id") == run) & (pl.col("target") == t)).sort("seed")
            if sub["seed"].to_list() != [0, 1, 2, 3, 4]:
                raise ValueError(f"missing selected configurations for {run} / {t}")
            options = {}
            for row in sub.iter_rows(named=True):
                options.setdefault(format_params(row), []).append(row["seed"])
            configs.extend(tex_row(name, [short, ",".join(map(str, seeds)), params]) for params, seeds in options.items())
    write("baseline-selected-configs", "\n".join([
        r"\begin{tabular}{llll}", r"\hline", tex_row("Model", ["Target", "Seeds", "Selected configuration"]),
        r"\hline", *configs, r"\hline", r"\end{tabular}",
    ]))


if __name__ == "__main__":
    main()
