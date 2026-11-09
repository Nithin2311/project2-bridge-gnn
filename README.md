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
| 02 | `02_features.py` — node feature matrix, bags, pooled features, artifacts | D |
| 03 | `03_directed_layer.py` — directed precedence edges (+ gap / day-one sensitivities), U-bh edges, 5 rewired + 1 shuffled null graphs; ~10 s | N (done) |
| 04 | `04_motifs.py` — triad census vs 100 directed rewirings, bridge middle shares, A -> bridge -> core chains; ~3 min | N (done) |
| 05 | `05_baselines.py` — non-graph baselines, GroupKFold, refit, test once | D |
| 06 | `06_gnn.py` — GCN / SAGE / GAT on patient subgraphs; GroupKFold selection, refit, test once; all experiment flags; ~5 min per (arch, target) | N (done) |
| 07 | `07_experiments.py` — drives the experiment grid | D |
| 08 | `08_evaluate.py` — bootstrap CIs, calibration, plots, tables | N |
| 09 | `09_pre2020_graph.py` — time-respecting rebuild + backtest | N |

## Branches

`main` is protected by convention: work on `nithin/<step>` or `dylan/<step>`, open a PR, the other reviews.
Cross-check branches (`nithin/xcheck-02`, ...) reimplement the other person's step and are compared, not merged.
