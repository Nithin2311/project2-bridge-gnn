# project2-bridge-gnn

CAI 5155 Network Analysis & ML with Graphs, Fall 2026. Project 2 (due **Fri Dec 4**): do bridge diagnoses and
graph structure predict a patient's entry into a disease community within 12 months? GCN / GraphSAGE / GAT on
patient subgraphs of the Project 1 comorbidity network, against tuned non-graph baselines, with the rubric's
five experiment families.

Team: Nithin Palyam, Dylan Girrens.

**Code only.** No EHRShot data, caches, predictions with patient ids, or checkpoints are ever committed.
`.gitignore` enforces it; keep it that way. The only CSVs allowed in git are the aggregate, category-level
artifacts under `outputs/artifacts/`.

## Read first

1. `docs/DESIGN.md` — the design (what, why, every decision).
2. `docs/DATA-CONTRACT.md` — the files that cross between steps and people; schemas are law.
3. `docs/INTERFACES.md` — the CLI contracts for steps 05 and 06 and the experiment grid.
4. Plans: `docs/plans/plan1-foundation.md` (Nithin) and
   `docs/plans/planD-dylan-features-baselines-driver.md` (Dylan).

## Layout

```
config/p2.yaml            every parameter with its reason
config/experiments.yaml   the experiment grid (IDs from the design, section 8)
src/bridgegnn/            library: config, contract, synth (fixtures), inputs, rows, extract, features, tabular, driver, ...
scripts/NN_*.py           pipeline steps in order; each writes to outputs/
tests/                    unit tests on synthetic data (no EHRShot needed)
outputs/                  gitignored except .gitkeep and artifacts/
docs/                     design, plans, contracts
```

## Setup

```bash
uv sync --extra dev --extra gnn                       # dev = pytest, ruff; gnn = torch + PyTorch Geometric (step 06)
export EHRSHOT_DATA="/path/to/FinalProject/data"      # the 13 CSVs; default ../FinalProject/data
uv run pytest -q                                      # synthetic tests, no data needed
uv run python scripts/00_check_inputs.py              # verifies Project 1 outputs and the DuckDB cache
```

Project 1 (`../project1-bridge-dx`) is a path dependency: its category map, label rules, range helper and null
models are imported (duckdb is pinned below 2.0 for that), and its `outputs/` folder is read as input. It is frozen;
nothing here writes to it. DuckDB runs in memory with views over the CSVs; no cache file is kept.

## Pipeline

| Step | Script | Owner |
|---|---|---|
| 00 | `00_check_inputs.py` — P1 outputs present, DuckDB tables, class balance | N (done) |
| 01 | `01_rows.py` — label rows + per-row code sets with windows and target masks; ~4 min | N (done) |
| 02 | `02_features.py` — node feature matrix, bags, pooled features, artifacts | D (done) |
| 03 | `03_directed_layer.py` — directed precedence edges (+ gap / day-one sensitivities), U-bh edges, 5 rewired + 1 shuffled null graphs; ~10 s | N (done) |
| 04 | `04_motifs.py` — triad census vs 100 directed rewirings, bridge middle shares, A -> bridge -> core chains; ~3 min | N (done) |
| 05 | `05_baselines.py` — non-graph baselines, GroupKFold, refit, test once | D (done) |
| 06 | `06_gnn.py` — GCN / SAGE / GAT on patient subgraphs; GroupKFold selection, refit, test once; all experiment flags; 25-135 s per (arch, target) on the M4 | N (done) |
| 07 | `07_experiments.py` — drives the experiment grid | D (done) |
| 08 | `08_evaluate.py` — patient-bootstrap CIs, calibration, F1 at the dev threshold, strata, paired deltas, ROC/PR/curve figures, LaTeX tables | N (done) |
| 09 | `09_pre2020_graph.py` — pre-2020 rebuild (P1 rules) and the B5 / backtest runs via step 06 | N (done) |
| all | `run_all.sh` — the labels above in order, logged, resumable; `collect_artifacts.py`, `compare_runs.py` | N (done) |

## Reproduce from scratch

Everything in `outputs/artifacts/` comes from one command on a fresh clone. Project 1 must sit at
`../project1-bridge-dx` with its frozen `outputs/` (run of 2026-10-23); the EHRShot CSVs live wherever `EHRSHOT_DATA`
points (default `../FinalProject/data`).

```bash
git clone https://github.com/Nithin2311/project2-bridge-gnn.git && cd project2-bridge-gnn
uv sync --extra dev --extra gnn
export EHRSHOT_DATA="/path/to/FinalProject/data"
./scripts/run_all.sh --check            # tests, dry run of the grid, input check (~2 min)
./scripts/run_all.sh                    # ~9 h on an M4: grid ~6 h, baselines ~1.5 h, evaluation < 15 min
uv run python scripts/compare_runs.py /path/to/reference/outputs/artifacts outputs/artifacts --md REPRODUCTION.md
```

`run_all.sh` runs the labels `00 01 02 03 04 05 06 07 09 08 08b macros collect` in order (05 = all tabular baselines through
the step 07 driver, 06 = the main GNN comparison, 07 = the remaining grid, 08b = the report fragments), appends
timestamped START/END lines per label to `outputs/logs/run_all.log`, stops at the first failure and resumes with
`--from <label>`. Approximate runtimes on an M4 (CPU): 00 2 min, 01 4 min, 02 1 min, 03 10 s, 04 3 min, 05 1.5 h,
06 15 min, 07 6 h, 09 11 min, 08 < 15 min.

`collect_artifacts.py` copies the aggregate set (`bridgegnn.artifacts.PATTERNS`: results, evaluation tables and
figures, layers, motifs, pre-2020 summary, node feature matrices) into `outputs/artifacts/`; it refuses parquet and
patient-list files, so nothing patient-level can land there. `compare_runs.py` then checks the new tree against a
reference tree table by table: `exact` means identical; `within_tol` is allowed only for torch-dependent numbers
(GNN rows of `results/gnn.csv` and the step 08 tables) up to 0.02; everything else, including the selected
hyperparameters in `params_json`, must match exactly. The fresh-clone comparison that produced the committed
artifacts is recorded in `outputs/artifacts/REPRODUCTION.md`.

### Only the tabular comparison

```bash
uv run python scripts/02_features.py
uv run python scripts/07_experiments.py --only-baselines
uv run python scripts/08_evaluate.py
```

If the Project 1 handoff and step 01 files are stored separately, pass their directories to step 02 using
`--p1-outputs` and `--rows-dir`. Put `rows.parquet` under the configured Project 2 output directory for step 07
patient subsampling and step 08 evaluation. Step 02 and step 05 also accept `--synthetic` for a local smoke run.
The driver accepts `--dry-run` to list resolved commands and `--ids` to rerun selected experiments.

The main model comparison uses the three primary targets; the main GNNs and four baselines also run on the
underpowered secondary target (`targets: all` in the grid). The report fragments in `report/` describe the
non-graph comparison and its reproducible inputs.

## Branches

`main` is protected by convention: work on `nithin/<step>` or `dylan/<step>`, open a PR, the other reviews.
Cross-check branches (`nithin/xcheck-02`, ...) reimplement the other person's step and are compared, not merged.
