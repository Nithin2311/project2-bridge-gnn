# Project 2 report contributions

Include `related-work.tex`, `baseline-methods.tex`, `baseline-results.tex`, and
`experiments-c1-d1.tex` in the corresponding sections of the joint LaTeX report.
Merge the citation keys in `dylan-refs.bib` into the report bibliography.
The table rows and result sentences read the generated fragments in this directory.
Use `baseline-selected-configs.tex` in the report appendix.

Run from the repository root after steps 05 and 07.

```bash
uv run python scripts/08_evaluate.py
uv run python scripts/08_report_baselines.py
```

The generator defaults to the archived GNN run aggregates for C1 and D1. Use
`--gnn-runs` with the results of a fresh full ablation grid. Use `--paired outputs/eval/tables/paired.csv`
with a step 08 paired table when GNN and baseline prediction files were scored
together. Aggregated GNN results alone support descriptive point comparisons.

The archived main comparison uses Nithin's five-seed GNN runs. The paired tables use a local
GraphSAGE seed-0 refit reconstructed with the same feature matrix and canonical test landmarks.
