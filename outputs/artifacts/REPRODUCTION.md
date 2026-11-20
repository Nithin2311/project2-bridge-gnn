# Reproduction record: fresh-clone re-run of 2026-11-19/20

Reference: the artifacts committed on `main` before this run (grid of 2026-11-09, baselines re-run 2026-11-13; reference
`params_json` predates the `index_from`/`index_to`/`test_index_from` flags). Re-run: a fresh `git clone` of
`https://github.com/Nithin2311/project2-bridge-gnn.git` into a sibling folder, `uv sync --extra dev --extra gnn`,
`uv run pytest -q` (175 passed), `./scripts/run_all.sh --check`, then `./scripts/run_all.sh` under `caffeinate`.

| | |
|---|---|
| Grid commit | `e98d56e01d4ef1873bf0d25d321959929e6a8937` (steps 00-09, 08, 08b, collect, macros) |
| B2 commit | `acb747b` pulled into the same clone afterwards; `07_experiments.py --ids B2`, then `run_all.sh --from 08` |
| Machine | Apple M4, 24 GB, macOS 26.6.2, CPU only |
| Environment | python 3.13.9, torch 2.15.0, torch_geometric 2.8.1, polars 2.1.0, scikit-learn 1.9.1, numpy 2.5.4, duckdb 1.5.6; uv 0.11.31; `uv.lock` sha256 27bb2476... |
| Project 1 input | `../project1-bridge-dx` at `b851488`, `outputs/` of 2026-10-23 (frozen) |
| Start / end | 2026-11-19T20:12:53Z / 2026-11-20T04:06:49Z (7 h 54 min); B2 pass 04:08-04:37Z |

## Step timings (`outputs/logs/run_all.log`)

| Label | Step | Elapsed | Exit |
|---|---|---|---|
| 00 | input check | 105 s | 0 |
| 01 | rows and row codes | 117 s | 0 |
| 02 | node features, tabular features, graph artifacts | 2 s | 0 |
| 03 | directed, BH and null layers | 10 s | 0 |
| 04 | motif census | 471 s | 0 |
| 05 | tabular baselines, 31 jobs (incl. metastatic and C1) | 1,891 s | 0 |
| 06 | main GNN comparison, 12 jobs (incl. metastatic) | 1,464 s | 0 |
| 07 | experiment grid, 87 jobs | 20,883 s | 0 |
| 09 | pre-2020 rebuild and B5 / backtest runs | 2,121 s | 0 |
| 08 | evaluation (bootstraps, strata, paired deltas, figures) | 1,372 s | 0 |
| 08b, collect, macros | fragments, artifact copy, macros | < 2 s | 0 |
| B2 | 3 jobs + `--from 08` | 7 min + 22 min | 0 |

## Comparison (`scripts/compare_runs.py`, tolerance 0.02 for torch rows)

Per-row check of the result tables (joined on run, target, seed):

| Table | Rows in both trees | Rows with identical AUROC and AUPRC | Rows that differ |
|---|---|---|---|
| `results/baselines.csv` | 135 | 105 | 30: `C1-25-hgb`, `C1-50-hgb` (all seeds) |
| `results/gnn.csv` | 540 | 510 | 30: `C1-25`, `C1-50` (all seeds) |

Every GNN row outside C1 reproduced **bit-for-bit** (max |delta AUROC| = 0 over 510 rows; the 0.02 tolerance was never
needed): torch on this CPU is deterministic given the seeds. Every tabular row outside C1 reproduced exactly. In the
evaluation tables (`runs`, `runs_agg`, `strata`, `paired`) no numeric column differs on any matched row outside C1.

**Expected differences, by design**

1. **C1 rows** (`C1-25`, `C1-50`, `C1-25-hgb`, `C1-50-hgb`). The reference GNN subsample lists were written by a shell loop on
   2026-11-09 before the step 07 driver existed; the reference boosting rows were re-run on 2026-11-13 on those same lists.
   The driver draws its own lists from the same seeds, and both families now train on identical lists
   (`tests/test_driver.py::test_write_subsamples_identical_across_families`), but those lists differ from the Nov 9
   ones, so every C1 row changes (max |delta AUROC| 0.12 on 25 % cohorts). The report's C1 numbers are the re-run's.
2. **New rows.** The secondary target `cancer_metastatic` (3 GNNs + 4 baselines, `targets: all`) and the B2 bipartite
   variant exist only in the re-run, so the row counts of every results and evaluation table grew; `compare_runs`
   reports these as "rows differ".
3. **Paired table.** The reference bootstrapped every run against both baselines (1,320 rows); the re-run pairs only
   the configured runs plus the explicit ablation pairings (225 rows, plus B2). The 120 pairs present in both agree.
4. **`params_json`.** Reference GNN rows lack the later-added null flags and record the cross-check feature directory
   (`features_xcheck`, bit-identical to step 02's matrix) instead of `outputs/features`; `compare_runs` ignores path
   keys and null-valued keys, so the only remaining `params_json` differences are the C1 rows' selected epochs.
5. **Removed file.** `layers/edges_directed_dayone.csv` (a stale duplicate) was removed from the reference before the run.

### Full comparison (nothing excluded)

| file | column | status | max abs diff | rows old | rows new | note |
|---|---|---|---|---|---|---|
| eval/tables/paired.csv | * | exceeds |  | 1320 | 270 | rows differ: 1320 old, 270 new, 120 matched on ['run_id', 'baseline', 'target', 'seed'] |
| eval/tables/reliability.csv | * | exceeds |  | 1350 | 1450 | rows differ: row counts (positional) |
| eval/tables/runs.csv | * | exceeds |  | 675 | 725 | rows differ: 675 old, 725 new, 672 matched on ['run_id', 'target', 'model', 'threshold_source', 'degenerate_threshold', 'seed'] |
| eval/tables/runs.csv | auroc | exceeds | 0.1163 | 675 | 725 |  |
| eval/tables/runs.csv | auprc | exceeds | 0.07556 | 675 | 725 |  |
| eval/tables/runs.csv | brier | exceeds | 0.03195 | 675 | 725 |  |
| eval/tables/runs.csv | slope | exceeds | 6.927 | 675 | 725 |  |
| eval/tables/runs.csv | intercept_joint | exceeds | 15.72 | 675 | 725 |  |
| eval/tables/runs.csv | citl | exceeds | 4.946 | 675 | 725 |  |
| eval/tables/runs.csv | tp | exceeds | 75 | 675 | 725 |  |
| eval/tables/runs.csv | fp | exceeds | 1354 | 675 | 725 |  |
| eval/tables/runs.csv | fn | exceeds | 75 | 675 | 725 |  |
| eval/tables/runs.csv | tn | exceeds | 1354 | 675 | 725 |  |
| eval/tables/runs.csv | precision | exceeds | 0.1291 | 675 | 725 |  |
| eval/tables/runs.csv | recall | exceeds | 0.8242 | 675 | 725 |  |
| eval/tables/runs.csv | f1 | exceeds | 0.07997 | 675 | 725 |  |
| eval/tables/runs.csv | threshold | exceeds | 0.1052 | 675 | 725 |  |
| eval/tables/runs.csv | pred_pos_rate | exceeds | 0.8156 | 675 | 725 |  |
| eval/tables/runs.csv | auroc_lo | exceeds | 0.09272 | 675 | 725 |  |
| eval/tables/runs.csv | auroc_hi | exceeds | 0.08921 | 675 | 725 |  |
| eval/tables/runs.csv | auprc_lo | exceeds | 0.01918 | 675 | 725 |  |
| eval/tables/runs.csv | auprc_hi | exceeds | 0.06169 | 675 | 725 |  |
| eval/tables/runs_agg.csv | * | exceeds |  | 135 | 145 | rows differ: 135 old, 145 new, 134 matched on ['run_id', 'target', 'model', 'any_degenerate_threshold', 'n_seeds', 'ci_seed'] |
| eval/tables/runs_agg.csv | auroc_mean | exceeds | 0.05017 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auroc_sd | exceeds | 0.03515 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auprc_mean | exceeds | 0.02872 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auprc_sd | exceeds | 0.01294 | 135 | 145 |  |
| eval/tables/runs_agg.csv | brier_mean | exceeds | 0.00876 | 135 | 145 |  |
| eval/tables/runs_agg.csv | brier_sd | exceeds | 0.00993 | 135 | 145 |  |
| eval/tables/runs_agg.csv | slope_mean | exceeds | 1.299 | 135 | 145 |  |
| eval/tables/runs_agg.csv | citl_mean | exceeds | 1.256 | 135 | 145 |  |
| eval/tables/runs_agg.csv | f1_mean | exceeds | 0.03523 | 135 | 145 |  |
| eval/tables/runs_agg.csv | f1_sd | exceeds | 0.02265 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auroc_seed | exceeds | 0.09051 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auprc_seed | exceeds | 0.0339 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auroc_lo | exceeds | 0.09272 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auroc_hi | exceeds | 0.08921 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auprc_lo | exceeds | 0.01918 | 135 | 145 |  |
| eval/tables/runs_agg.csv | auprc_hi | exceeds | 0.06169 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | * | exceeds |  | 135 | 145 | rows differ: 135 old, 145 new, 134 matched on ['run_id', 'target', 'model', 'any_degenerate_threshold', 'n_seeds', 'ci_seed'] |
| eval/tables/runs_agg_rounded.csv | auroc_mean | exceeds | 0.05 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auroc_sd | exceeds | 0.035 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auprc_mean | exceeds | 0.028 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auprc_sd | exceeds | 0.013 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | brier_mean | exceeds | 0.009 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | brier_sd | exceeds | 0.01 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | slope_mean | exceeds | 1.299 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | citl_mean | exceeds | 1.256 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | f1_mean | exceeds | 0.035 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | f1_sd | exceeds | 0.023 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auroc_seed | exceeds | 0.091 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auprc_seed | exceeds | 0.034 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auroc_lo | exceeds | 0.092 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auroc_hi | exceeds | 0.089 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auprc_lo | exceeds | 0.019 | 135 | 145 |  |
| eval/tables/runs_agg_rounded.csv | auprc_hi | exceeds | 0.062 | 135 | 145 |  |
| eval/tables/strata.csv | * | exceeds |  | 1543 | 1662 | rows differ: 1543 old, 1662 new, 1543 matched on ['stratum', 'level', 'run_id', 'target', 'seed'] |
| eval/tables/strata.csv | auroc | exceeds | 0.2189 | 1543 | 1662 |  |
| eval/tables/strata.csv | auprc | exceeds | 0.2646 | 1543 | 1662 |  |
| results/baselines.csv | * | exceeds |  | 135 | 155 | rows differ: 135 old, 155 new, 135 matched on ['run_id', 'family', 'model', 'target', 'layer', 'features', 'seed'] |
| results/baselines.csv | params_json | exceeds | 28 | 135 | 155 | 28 cells differ |
| results/baselines.csv | dev_oof_auprc | exceeds | 0.1505 | 135 | 155 |  |
| results/baselines.csv | test_auroc | exceeds | 0.1163 | 135 | 155 |  |
| results/baselines.csv | test_auprc | exceeds | 0.07021 | 135 | 155 |  |
| results/baselines.csv | test_brier | exceeds | 0.03195 | 135 | 155 |  |
| results/gnn.csv | * | exceeds |  | 540 | 570 | rows differ: 540 old, 570 new, 540 matched on ['run_id', 'family', 'model', 'target', 'layer', 'features', 'seed'] |
| results/gnn.csv | params_json | exceeds | 29 | 540 | 570 | 29 cells differ |
| results/gnn.csv | dev_oof_auprc | exceeds | 0.1959 | 540 | 570 |  |
| results/gnn.csv | test_auroc | exceeds | 0.08989 | 540 | 570 |  |
| results/gnn.csv | test_auprc | exceeds | 0.07556 | 540 | 570 |  |
| results/gnn.csv | test_brier | exceeds | 0.03 | 540 | 570 |  |

### Comparison with `--exclude-runs C1- B2`

| file | column | status | max abs diff | rows old | rows new | note |
|---|---|---|---|---|---|---|
| eval/tables/paired.csv | * | exceeds |  | 1200 | 225 | rows differ: 1200 old, 225 new, 120 matched on ['run_id', 'baseline', 'target', 'seed'] |
| eval/tables/reliability.csv | * | exceeds |  | 1230 | 1300 | rows differ: row counts (positional) |
| eval/tables/runs.csv | * | exceeds |  | 615 | 650 | rows differ: 615 old, 650 new, 615 matched on ['run_id', 'target', 'model', 'threshold_source', 'degenerate_threshold', 'seed'] |
| eval/tables/runs_agg.csv | * | exceeds |  | 123 | 130 | rows differ: 123 old, 130 new, 123 matched on ['run_id', 'target', 'model', 'any_degenerate_threshold', 'n_seeds', 'ci_seed'] |
| eval/tables/runs_agg_rounded.csv | * | exceeds |  | 123 | 130 | rows differ: 123 old, 130 new, 123 matched on ['run_id', 'target', 'model', 'any_degenerate_threshold', 'n_seeds', 'ci_seed'] |
| eval/tables/strata.csv | * | exceeds |  | 1403 | 1487 | rows differ: 1403 old, 1487 new, 1403 matched on ['stratum', 'level', 'run_id', 'target', 'seed'] |
| results/baselines.csv | * | exceeds |  | 105 | 125 | rows differ: 105 old, 125 new, 105 matched on ['run_id', 'family', 'model', 'target', 'layer', 'features', 'seed'] |
| results/gnn.csv | * | exceeds |  | 510 | 525 | rows differ: 510 old, 525 new, 510 matched on ['run_id', 'family', 'model', 'target', 'layer', 'features', 'seed'] |

All eight remaining rows are row-count differences from the new metastatic and pairing rows; every matched cell is exact.

## What the submitted artifacts are

`outputs/artifacts/` on `main` is the re-run tree (grid at `e98d56e`, B2 at `acb747b`), copied with
`collect_artifacts.py`; `features_xcheck/` is the one-off cross-check kept from 2026-11-13 and is not produced by
`run_all.sh`. The report's fragments and `macros.tex` were regenerated from this tree.


## Second pass (2026-11-20, after the review panel): controls and seed-pooled evaluation

In the same clone, `git pull` to the panel-fix branch (`632247d`), then `07_experiments.py --ids B-cov_lr D1-empty`
(covariates-only logistic regression on all targets; edgeless GraphSAGE on the primary targets), `10_cohort_tables.py`
and `run_all.sh --from 08` (evaluation with seed-pooled paired rows and the sex-gap bootstrap, fragments, collect,
macros). Step 08 took 24 min. Nothing upstream of step 08 was re-run, so every previously reported row is unchanged;
the comparison against the first-pass tree, excluding the runs that only exist in the second pass and the C1 rows:

| file | column | status | max abs diff | rows old | rows new | note |
|---|---|---|---|---|---|---|
| eval/tables/paired.csv | * | exceeds |  | 225 | 516 | rows differ: 225 old, 516 new, 225 matched on ['run_id', 'baseline', 'target', 'seed'] |
| eval/tables/sex_gap.csv | * | missing_file |  |  |  | absent in old |

The remaining rows are the added runs (new rows in the results and evaluation tables) and the paired table, which
now carries one seed-pooled row (`seed = -1`) per pairing and pairs against five references instead of two.
