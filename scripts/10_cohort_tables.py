"""Step 10: aggregate cohort tables for the report (no patient-level output).

Writes outputs/cohort/{cohort,triggering_codes,subgraph_sizes,core_members,bridge_carriers}.csv from the step 01 rows,
the step 01 row codes and the frozen Project 1 outputs, plus LaTeX fragments in report/ (cohort-table.tex,
trigger-table.tex, core-table.tex, bridge-carriers-table.tex). Rerunnable: `uv run python scripts/10_cohort_tables.py`.
"""
from __future__ import annotations

import time

import polars as pl

from bridgegnn import cohort, config, contract, nodefeat

SHORT = {"cardiometabolic": "CM", "cancer_hematologic": "Heme", "mental_health": "MH", "cancer_metastatic": "Meta"}


def cell(n: float) -> str:
    """Counts below five are shown as '<5' (small-cell suppression for the published fragments)."""
    return "$<$5" if n < 5 else f"{n:,.0f}"


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main() -> None:
    t0 = time.time()
    cfg = config.load()
    out = config.outputs_dir(cfg)
    p1 = config.p1_outputs(cfg)
    (out / "cohort").mkdir(parents=True, exist_ok=True)
    rep = config.REPO_ROOT / "report"
    rep.mkdir(exist_ok=True)
    targets = list(cfg["task"]["primary_targets"]) + list(cfg["task"]["secondary_targets"])
    rows = contract.read(out / "rows.parquet", "rows")
    row_codes = contract.read(out / "row_codes.parquet", "row_codes")
    tdef = pl.read_csv(p1 / "target_definitions.csv")
    nf = pl.read_csv(p1 / "p2" / "node_features.csv")
    bridges = pl.read_csv(p1 / "bridges.csv")
    first = pl.read_parquet(p1 / "patient_category.parquet").select("person_id", "category", "first_date")
    horizon_days = round(int(cfg["task"]["horizon_months"]) * 30.4375) + 1

    core = {t: tdef.filter((pl.col("target") == t) & pl.col("in_core"))["category"].to_list() for t in targets}
    coh = cohort.cohort_table(rows, tuple(cfg["split"]["dev"]))
    trig = cohort.triggering_codes(rows.filter(pl.col("positive")), first, core, horizon_days)
    sizes = cohort.subgraph_sizes(rows, row_codes, targets)
    members = cohort.core_members(tdef, nf)
    flags = {t: sorted(nodefeat.bflag_codes(nf, bridges, t)) for t in targets}
    carriers = cohort.bridge_carriers(rows, row_codes, flags)
    for name, df in [("cohort", coh), ("triggering_codes", trig), ("subgraph_sizes", sizes), ("core_members", members), ("bridge_carriers", carriers)]:
        df.write_csv(out / "cohort" / f"{name}.csv")

    # ---- LaTeX fragments ----
    names = dict(zip(nf["category"].to_list(), nf["name"].to_list())) if "name" in nf.columns else {}
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\hline",
             r"Target & Dev rows & Dev patients & Dev pos. & Test rows & Test patients & Test pos. & Female (test) & Age 65+ (test) & Median nodes \\", r"\hline"]
    for r in coh.iter_rows(named=True):
        sz = sizes.filter(pl.col("target") == r["target"])
        med = f"{sz['median_nodes'][0]:.0f} [{sz['q1_nodes'][0]:.0f}--{sz['q3_nodes'][0]:.0f}]" if sz.height else "--"
        lines.append(f"{SHORT.get(r['target'], r['target'])} & {r['n_rows_dev']:,} & {r['n_patients_dev']:,} & {r['n_pos_dev']:,} & {r['n_rows_test']:,} & "
                     f"{r['n_patients_test']:,} & {r['n_pos_test']:,} & {100 * r['test_female_share']:.0f}\\% & {100 * r['test_age65_share']:.0f}\\% & {med} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (rep / "cohort-table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrr}", r"\hline", r"Target & Triggering code (first core code in the horizon; same-day ties share credit) & Positives & Share \\", r"\hline"]
    for t in targets:
        sub = trig.filter(pl.col("target") == t).head(6)
        for i, r in enumerate(sub.iter_rows(named=True)):
            lab = SHORT.get(t, t) if i == 0 else ""
            lines.append(f"{lab} & {esc(r['category'])} {esc(names.get(r['category'], ''))} & {cell(r['n'])} & {100 * r['share']:.0f}\\% \\\\")
        rest = trig.filter(pl.col("target") == t).slice(6)
        if rest.height:
            lines.append(f" & other ({rest.height} codes) & {cell(float(rest['n'].sum()))} & {100 * rest['share'].sum():.0f}\\% \\\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (rep / "trigger-table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llp{0.62\textwidth}}", r"\hline", r"Target & Members & Core codes with development prevalence (chapter-block members marked $^{b}$) \\", r"\hline"]
    for t in targets:
        sub = members.filter(pl.col("target") == t)
        cells = [f"{esc(r['category'])} {100 * (r['prevalence'] or 0):.0f}\\%{'$^{b}$' if r['in_chapter_block'] else ''}" for r in sub.iter_rows(named=True)]
        lines.append(f"{SHORT.get(t, t)} & {sub.height} & {', '.join(cells)} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (rep / "core-table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrrr}", r"\hline", r"Target & Bridge code & Eligible rows carrying it & Positive rows carrying it & Eligible rows \\", r"\hline"]
    for r in carriers.iter_rows(named=True):
        lines.append(f"{SHORT.get(r['target'], r['target'])} & {esc(r['category'])} {esc(names.get(r['category'], ''))} & {cell(r['n_rows'])} & {cell(r['n_pos_rows'])} & {r['n_rows_total']:,} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (rep / "bridge-carriers-table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'cohort'} (5 tables) and 4 fragments in {rep}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
