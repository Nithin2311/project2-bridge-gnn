# Project 2 report contributions (Dylan)

Include `related-work.tex`, `baseline-methods.tex`, `baseline-results.tex`, and `experiments-c1-d1.tex` in the
corresponding sections of the joint LaTeX report; `baseline-ci-table.tex` and `baseline-selected-configs.tex` go in the
appendix. Merge `dylan-refs.bib` into the report bibliography. The `*-rows.tex` and `*-summary.tex` fragments are
GENERATED; never edit them by hand.

Regenerate (repository root) after steps 05, 06/07 and 08 have run, so that `outputs/eval/tables/runs_agg.csv` holds
every run and `outputs/eval/tables/paired.csv` holds the GNN-vs-baseline deltas computed on the archived GNN
predictions and the baseline predictions together:

```bash
uv run python scripts/08_evaluate.py
uv run python scripts/08_report_baselines.py
```

Committed copies of the tables live under `outputs/artifacts/eval/tables/`.
