"""Generate the LaTeX result fragments from step 08 aggregates and step 05 configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from bridgegnn import config, contract

TARGETS = {"cardiometabolic": "CM", "cancer_hematologic": "Heme", "mental_health": "MH"}
BASELINES = {
    "B-prevalence": "Prevalence", "B-lr": "Logistic regression", "B-hgb": "Graph-code boosting",
    "B-hgb_all": "All-code boosting", "B-mlp": "MLP", "B-mlp_pool": "Pooled MLP",
    "B-bridge_lr": "Bridge logistic regression",
}
REWIRES = [f"D1-rewire{i}" for i in range(5)]
SCALE = {
    "C1-25": r"GraphSAGE, 25\%", "C1-50": r"GraphSAGE, 50\%", "G-sage": r"GraphSAGE, 100\%",
    "C1-25-hgb": r"Boosting, 25\%", "C1-50-hgb": r"Boosting, 50\%", "B-hgb": r"Boosting, 100\%",
}


def tex_row(label: str, fields: list[str]) -> str:
    return " & ".join([label, *fields]) + r" \\"


def table_map(file: Path) -> dict:
    return {(r["run_id"], r["target"]): r for r in pl.read_csv(file).iter_rows(named=True)}


def format_params(row: dict) -> str:
    p = json.loads(row["params_json"])
    model = row["model"]
    if model == "prevalence":
        return "Development prior"
    if model in {"lr", "bridge_lr"}:
        return rf"$C={p['C']:g}$"
    if model in {"hgb", "hgb_all"}:
        return rf"$\eta={p['learning_rate']:g}$, {p['max_leaf_nodes']} leaves"
    widths = "--".join(map(str, p["hidden_layer_sizes"]))
    alpha = r"10^{-4}" if p["alpha"] == 1e-4 else r"10^{-3}"
    return rf"{widths} units, $\alpha={alpha}$, {p['refit_epochs']} epochs"


def main() -> None:
    root = config.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, default=root / "outputs/eval/tables/runs_agg.csv")
    parser.add_argument("--gnn-runs", type=Path, default=root / "outputs/artifacts/eval/tables/runs_agg.csv")
    parser.add_argument("--baseline-results", type=Path, default=root / "outputs/results/baselines.csv")
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--out", type=Path, default=root / "report")
    args = parser.parse_args()
    stats = table_map(args.gnn_runs)
    stats.update({key: row for key, row in table_map(args.baseline_runs).items()
                  if key[0] in BASELINES or key[0] in {"C1-25-hgb", "C1-50-hgb"}})
    rows = contract.read(args.baseline_results, "results")
    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, content):
        (args.out / f"{name}.tex").write_text(content + "\n", encoding="utf-8")

    def ap(run, target):
        return stats[(run, target)]["auprc_mean"]

    def number(run, target):
        row = stats[(run, target)]
        return f"{row['auprc_mean']:.3f}" + r"\(\pm\)" + f"{row['auprc_sd'] or 0.0:.3f}"

    for run in {*BASELINES, *SCALE, *REWIRES, "D1-shuffle", "G-gcn", "G-gat"}:
        for target in TARGETS:
            row = stats.get((run, target))
            if row is None or row["n_seeds"] != 5:
                raise ValueError(f"missing five-seed aggregate for {run} / {target}")

    for name, mapping in [("baseline-ap-rows", BASELINES), ("scale-ap-rows", SCALE)]:
        write(name, "\n".join(tex_row(label, [number(run, t) for t in TARGETS])
                              for run, label in mapping.items()))
    ranges = {t: (min(ap(run, t) for run in REWIRES), max(ap(run, t) for run in REWIRES)) for t in TARGETS}
    write("null-ap-rows", "\n".join([
        tex_row("Layer U", [number("G-sage", t) for t in TARGETS]),
        tex_row("Degree-preserving rewires", [f"{lo:.3f}--{hi:.3f}" for lo, hi in ranges.values()]),
        tex_row("Edge shuffle", [number("D1-shuffle", t) for t in TARGETS]),
    ]))

    notes = []
    for t, short in TARGETS.items():
        winner = max(BASELINES, key=lambda r: ap(r, t))
        strongest_gnn = max(ap(r, t) for r in ("G-gcn", "G-sage", "G-gat"))
        notes.append(f"For {short}, {BASELINES[winner].lower()} had the highest observed baseline mean "
                     f"AUPRC ({ap(winner, t):.3f}), and the highest GNN mean was {strongest_gnn:.3f}.")
    if args.paired:
        paired = pl.read_csv(args.paired).filter(
            (pl.col("run_id") == "G-sage") & (pl.col("seed") == 0)
            & pl.col("baseline").is_in(["B-hgb_all", "B-mlp_pool"]))
        if paired.height == 6:
            notes.append("Seed-0 differences for the jointly evaluated GraphSAGE run are in the paired "
                         "evaluation table, with 95\\% patient-bootstrap intervals.")
    write("baseline-results-summary", " ".join(notes))
    scale_notes = []
    for label, full, quarter in [("GraphSAGE", "G-sage", "C1-25"), ("boosting", "B-hgb", "C1-25-hgb")]:
        diffs = ", ".join(f"{ap(full, t) - ap(quarter, t):+.3f}" for t in TARGETS)
        scale_notes.append(f"The full-minus-quarter differences for {label} were {diffs}, in target order.")
    write("scale-summary", " ".join(scale_notes) + " Changes reflect training cohort size and event mix.")
    null_notes = []
    for t, short in TARGETS.items():
        lo, hi = ranges[t]
        real, shuffled = ap("G-sage", t), ap("D1-shuffle", t)
        relation = "above" if real > hi else "below" if real < lo else "within"
        null_notes.append(f"For {short}, the observed-edge mean ({real:.3f}) was {relation} the rewire range "
                          f"({lo:.3f}--{hi:.3f}), and the shuffled-edge mean was {shuffled:.3f}.")
    write("null-summary", " ".join(null_notes) + " These contrasts give descriptive evidence about edge topology.")

    configs = []
    for run, label in BASELINES.items():
        for t, short in TARGETS.items():
            subgroup = rows.filter((pl.col("run_id") == run) & (pl.col("target") == t)).sort("seed")
            if subgroup["seed"].to_list() != [0, 1, 2, 3, 4]:
                raise ValueError(f"missing selected configurations for {run} / {t}")
            options = {}
            for row in subgroup.iter_rows(named=True):
                options.setdefault(format_params(row), []).append(row["seed"])
            configs.extend(tex_row(label, [short, ",".join(map(str, seeds)), params])
                           for params, seeds in options.items())
    write("baseline-selected-configs", "\n".join([
        r"\begin{tabular}{llll}", r"\hline", tex_row("Model", ["Target", "Seeds", "Selected configuration"]),
        r"\hline", *configs, r"\hline", r"\end{tabular}",
    ]))


if __name__ == "__main__":
    main()
