# Interfaces: the CLIs steps call on each other

Two scripts are called by other people's code: step 05 (Dylan) and step 06 (Nithin). Their command lines are
fixed here so the experiment driver (step 07, Dylan) can be written and dry-run before step 06 exists. Both
scripts read `config/p2.yaml`, write `predictions` parquet files and append `results` rows per the contract,
exit 0 on success, and print one JSON line per seed on stdout:

```json
{"run_id": "A1a", "target": "cardiometabolic", "seed": 0, "dev_oof_auprc": 0.21, "test_auroc": 0.71, "test_auprc": 0.24}
```

## Step 05 — non-graph baselines (Plan D, Dylan)

```
uv run python scripts/05_baselines.py --run-id B-lr --target cardiometabolic --model lr [--seeds 0 1 2 3 4]
                                       [--subsample-file outputs/subsamples/C1-25_seed0.txt] [--synthetic]
```

| flag | values | default |
|---|---|---|
| `--run-id` | any string; baselines use `B-<model>` | required |
| `--target` | cardiometabolic, cancer_hematologic, mental_health, cancer_metastatic | required |
| `--model` | prevalence, lr, hgb, hgb_all, mlp, mlp_pool, bridge_lr | required |
| `--seeds` | integers | `cfg.seeds` |
| `--subsample-file` | path template with a literal `{seed}`, e.g. `outputs/subsamples/C1-25_seed{seed}.txt`; for each seed the script reads that file (one development `person_id` per line) and uses only those patients' development rows for selection and refit; test untouched. The driver writes these files with `driver.subsample_patients` so step 05 and step 06 train on identical subsets | none |
| `--synthetic` | run the same code path on `synth.make(0)` in a temp dir | off |

Outputs: `outputs/predictions/{run_id}_{target}_seed{seed}.parquet`, a row per seed appended to
`outputs/results/baselines.csv`, log at `outputs/logs/05_{target}_{model}.log`. `family` in the results row is
looked up by `run_id` in `config/experiments.yaml` (`baseline` when the id is not listed, e.g. `B-lr`). Append
= read existing CSV if present, concat, `contract.write`. stdout carries only the JSON lines.

## Step 06 — GNN on patient subgraphs (Plan 3, Nithin)

```
uv run python scripts/06_gnn.py --run-id A1a --target cardiometabolic --arch sage --layer U
    --features CODE,DEMO,UTIL,TEMP,STRUCT,BRIDGE --lookback 1095 --expand 0 --weighted 1
    --depth 2 --agg mean --heads 4 --hidden 64 --seeds 0 1 2 3 4 --null-graph none
    [--drop-codes none] [--subsample-file outputs/subsamples/C1-25_seed{seed}.txt] [--dry-run]
```

| flag | values | default |
|---|---|---|
| `--run-id` | experiment id from `config/experiments.yaml`, or `G-<arch>` for the main comparison | required |
| `--target` | as above | required |
| `--arch` | gcn, sage, gat | required |
| `--layer` | U (P1 main graph), U-bb (top-10 backbone), U-bh (BH-FDR edges), D (directed precedence), N (null graph; see `--null-graph`) | U |
| `--features` | comma-joined subset of CODE, DEMO, UTIL, TEMP, STRUCT, BRIDGE | all six |
| `--lookback` | 365, 1095, 0 (all) | 1095 |
| `--expand` | 0 or 1: add 1-hop global neighbours as context nodes | 0 |
| `--weighted` | 0 or 1: use phi edge weights where the layer supports it | 1 |
| `--depth` | 2 or 3 | 2 |
| `--agg` | mean, max (GraphSAGE only) | mean |
| `--heads` | 2, 4, 8 (GAT only) | 4 |
| `--hidden` | 32, 64, 128 | 64 |
| `--seeds` | integers | `cfg.seeds` |
| `--subsample-file` | as step 05 (same `{seed}` template; the only subsampling mechanism, so both steps see identical subsets) | none |
| `--null-graph` | none, rewire0 .. rewire4, shuffle (requires `--layer N`) | none |
| `--drop-codes` | none, bridge (remove the target's B-flag codes from the subgraph and the UTIL code counts) | none |
| `--node-features-dir` | directory with `node_index.csv`, `node_x.npy`, `feature_groups.json`; default `outputs/features/` (step 02) if present, else built into `outputs/features_xcheck/` from Project 1 files | auto |
| `--layer-file` | graphml path; overrides `--layer`; recorded as `layer: file:<stem>`; direction read from the file | none |
| `--index-from`, `--index-to` | inclusive index-date window applied to every row; when `--test-index-from` is also given, `--index-to` bounds development rows only | none |
| `--test-index-from` | inclusive lower bound on test rows only (the temporal backtest's test window) | none |
| `--quick` | smoke run with `gnn.quick` epochs and patience | off |
| `--dry-run` | print the resolved configuration (flags, model kwargs, grid, node-feature dir, family) as JSON and exit 0 without importing torch | off |

Protocol: hyperparameters (`gnn.lr_grid x gnn.weight_decay_grid`) are selected once per (run_id, target) on
`gnn.selection_seed` by GroupKFold (`split.cv_folds`) over development rows, early stopping on fold-validation
AUPRC; then each seed refits on all development rows for the mean best epoch count and predicts test once. With
`--subsample-file`, selection and refit are repeated per seed on that seed's patient subset. CODE = learnable
node-id embedding; CHAPTER and COMMUNITY node columns travel with STRUCT. On Layer D, GCN receives the
symmetrised edge set (`symmetrise: true` in `params_json`); SAGE and GAT use the directed edges.

Outputs: `outputs/predictions/{run_id}_{target}_seed{seed}.parquet` (development rows carry the selection
run's out-of-fold `p`, test rows the seed's refit), a row per seed upserted into `outputs/results/gnn.csv`
(keyed by run_id, target, seed), curves at `outputs/logs/06_{run_id}_{target}_seed{seed}.json` (per-fold
selection curves and the refit curve), log at `outputs/logs/06_{run_id}_{target}.log`. `family` in the results
row is looked up by run_id in `config/experiments.yaml`, `gnn` when absent.

## Step 07 — experiment driver (Plan D, Dylan)

```
uv run python scripts/07_experiments.py [--ids A1a B1 ...] [--dry-run] [--only-baselines]
```

Reads `config/experiments.yaml`. Each entry:

```yaml
- id: A1a
  family: A
  base: gnn            # gnn -> scripts/06_gnn.py ; baseline -> scripts/05_baselines.py
  targets: primary     # or an explicit list
  overrides: {features: "CODE,DEMO,UTIL,TEMP,STRUCT"}   # merged over the step's defaults as CLI flags
  note: "full features minus BRIDGE (features only)"
```

The driver expands `targets: primary` from `p2.yaml`, injects `default_arch` as `--arch` when an experiment does
not override it, writes `outputs/subsamples/{id}_seed{seed}.txt` for `subsample` overrides and passes the
template via `--subsample-file` to both steps, runs
one subprocess per (experiment, target), logs to `outputs/logs/07_{id}_{target}.log`, continues on failure,
and summarises exit codes. `--dry-run` prints the commands and exits 0 even when the target script does not
exist yet.


## Step 08 — evaluation (Nithin)

```
uv run python scripts/08_evaluate.py [--n-bootstrap N]
```
Reads every `outputs/predictions/*.parquet` and `outputs/rows.parquet`; writes `outputs/eval/tables/{runs,runs_agg,
reliability,strata,paired}.csv`, `runs_agg.tex`, `paired.tex`, and `outputs/eval/figures/{roc_pr,reliability}_{target}.pdf`,
`curves_<run>_<target>.pdf`. Paired deltas appear for every non-baseline run against each of
`eval.baselines_for_pairing` whose predictions exist (step 05). Strata: sex, age band, subgraph-size bin, index year;
fewer than `eval.min_stratum_pos` positives prints n/a.

## Step 09 — pre-2020 rebuild and backtest (Nithin)

```
uv run python scripts/09_pre2020_graph.py [--skip-runs] [--arch sage] [--targets ...]
```
Builds `outputs/pre2020/` (graph, edges, cores, bridges, node features in nodefeat format, summary vs Layer U) from
development records before `rebuild.cutoff`, then calls step 06 for `B5-U`, `B5-P`, `BT-U`, `BT-P` per primary target.
