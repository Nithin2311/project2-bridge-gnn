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
| `--subsample-file` | path to a text file with one development `person_id` per line; only those patients' development rows are used for selection and refit; test untouched | none |
| `--synthetic` | run the same code path on `synth.make(0)` in a temp dir | off |

Outputs: `outputs/predictions/{run_id}_{target}_seed{seed}.parquet`, a row per seed appended to
`outputs/results/baselines.csv`, log at `outputs/logs/05_{target}_{model}.log`.

## Step 06 — GNN on patient subgraphs (Plan 3, Nithin)

```
uv run python scripts/06_gnn.py --run-id A1a --target cardiometabolic --arch sage --layer U
    --features CODE,DEMO,UTIL,TEMP,STRUCT,BRIDGE --lookback 1095 --expand 0 --weighted 1
    --depth 2 --agg mean --heads 4 --hidden 64 --seeds 0 1 2 3 4 --subsample 1.0 --null-graph none
    [--drop-codes none] [--subsample-file PATH] [--dry-run]
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
| `--subsample` | fraction of development patients in (0, 1]; patient-level, seeded | 1.0 |
| `--subsample-file` | as step 05; overrides `--subsample` | none |
| `--null-graph` | none, rewire0 .. rewire4, shuffle (requires `--layer N`) | none |
| `--drop-codes` | none, bridge (remove the target's B-flag codes from the subgraph and the bag) | none |
| `--dry-run` | print the resolved configuration and exit 0 without training | off |

Outputs: `outputs/predictions/{run_id}_{target}_seed{seed}.parquet`, a row per seed appended to
`outputs/results/gnn.csv`, curves at `outputs/logs/06_{run_id}_{target}_seed{seed}.json`, log at
`outputs/logs/06_{run_id}_{target}.log`. `family` in the results row is `gnn` for `G-*` runs and the
experiment's family letter otherwise.

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

The driver expands `targets: primary` from `p2.yaml`, writes subsample files for `subsample` overrides, runs
one subprocess per (experiment, target), logs to `outputs/logs/07_{id}_{target}.log`, continues on failure,
and summarises exit codes. `--dry-run` prints the commands and exits 0 even when the target script does not
exist yet.
