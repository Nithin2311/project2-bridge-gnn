# Plan 5: Pipeline closing (leftovers, one-command run, comparison, fresh-clone re-run)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the pipeline leftovers (L1-L6 of the closing design), add a one-command reproduction and a run
comparison tool, then launch the fresh-clone re-run whose outputs become the submitted numbers.

**Architecture:** Config-only additions for the secondary target (`targets: all` in the grid) and the pairing
restriction (`eval.paired_gnn_runs`); two small library modules (`compare.py`, `artifacts.py`) with CLIs; one bash
orchestrator `scripts/run_all.sh` that chains steps 00-09 through the existing step 07 driver. No change to any
model, feature or label code.

**Tech Stack:** Python 3.11, polars, numpy, pytest, bash; uv; existing `bridgegnn` package.

**Spec:** `docs/CLOSING-DESIGN.md` sections 5.1 and 5.2 (plus 4 and 7).

## Global Constraints

- Never commit patient-level files: `rows.parquet`, `row_codes.parquet`, `predictions/*.parquet`, `subsamples/*.txt`.
  Only aggregates under `outputs/artifacts/` (`.csv .tex .json .pdf .md .npy`) are tracked.
- Project 1 (`../project1-bridge-dx`) is frozen; nothing writes to it.
- `duckdb>=1.1,<2`; `uv sync --extra dev --extra gnn`; `uv run ruff check .` clean; `uv run pytest -q` green.
- Results rows are upserted by (run_id, target, seed); re-running a step never duplicates rows.
- Deterministic steps (01-05, 09 rebuild, tabular baselines) must reproduce exactly; torch runs within 0.02 AUROC
  per (run, target, seed) with identical selected hyperparameters.
- Commit messages: imperative subject; no tooling markers in the repo (shared with Dylan).

## Review Focus

Failure modes the spec implies that no task's tests covered until this list added them:

1. A grid entry with `targets: all` is run with `--only-baselines`: metastatic baseline jobs must be included and
   primary-only entries unaffected. (Task 2 test `test_expand_targets_all_includes_secondary`.)
2. `eval.paired_gnn_runs` absent from an older config: `_pairings` must fall back to the previous behaviour
   (every non-baseline run paired). (Task 1 test `test_pairings_fallback_when_unrestricted`.)
3. `compare_dirs` meets a CSV present in one tree only, or a column present in one file only: reported as a
   finding, never a crash or a silent skip. (Task 4 test `test_compare_reports_missing_file_and_column`.)
4. `compare_dirs` on `params_json` strings: a changed selected hyperparameter is an "exceeds" even when metrics
   agree. (Task 4 test `test_compare_params_json_must_match_exactly`.)
5. `run_all.sh` resumed with `--from 06` after a failure must not rerun 01 (which would rewrite `rows.parquet`
   and invalidate archived subsample lists). (Task 3 step "resume check", shell-level.)
6. `C1-25` (GNN) and `C1-25-hgb` (boosting) get different subsample file names from the driver; the report claims both
   families trained on identical patient subsets. That holds only because `subsample_patients` depends on (ids, fraction,
   seed) alone. (Task 2 test `test_same_fraction_and_seed_give_identical_subsets` pins it.)

---

### Task 1: Restrict paired bootstraps to the configured GNN runs and add the ablation pairings

**Files:**
- Modify: `config/p2.yaml` (eval block, lines 74-80)
- Modify: `src/bridgegnn/eval/report.py:44-50` (`_pairings`)
- Modify: `docs/INTERFACES.md` step 08 paragraph
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Produces: `_pairings(cfg_eval: dict, run_ids: set[str]) -> list[tuple[str, str]]` (unchanged signature). With
  `cfg_eval["paired_gnn_runs"]` present, baseline pairings are built only for runs in that list; `paired_runs`
  pairs are added as before. Without the key, behaviour is unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_eval_report.py`)

```python
def test_pairings_restricted_to_configured_gnn_runs():
    e = {"baselines_for_pairing": ["B-hgb_all", "B-mlp_pool"], "paired_gnn_runs": ["G-sage"],
         "paired_runs": [["A1a", "G-sage"]]}
    pairs = report._pairings(e, {"G-sage", "G-gcn", "A1a", "E3-h32", "B-hgb_all", "B-mlp_pool"})
    assert pairs == [("G-sage", "B-hgb_all"), ("G-sage", "B-mlp_pool"), ("A1a", "G-sage")]

def test_pairings_fallback_when_unrestricted():
    e = {"baselines_for_pairing": ["B-hgb_all"], "paired_runs": []}
    pairs = report._pairings(e, {"G-sage", "E3-h32", "B-hgb_all"})
    assert pairs == [("E3-h32", "B-hgb_all"), ("G-sage", "B-hgb_all")]
```

- [ ] **Step 2: Run them**: `uv run pytest tests/test_eval_report.py -k pairings -v` → first FAILS (E3-h32 and G-gcn paired), second PASSES already.

- [ ] **Step 3: Implement** in `_pairings`: `gnn_runs = cfg_eval.get("paired_gnn_runs")`; the comprehension over
  `sorted(run_ids)` keeps `r` only when `gnn_runs is None or r in gnn_runs`. Order: baseline pairs first (sorted run,
  then baseline order), then `paired_runs` in config order.

- [ ] **Step 4: Config** (`config/p2.yaml` eval block):

```yaml
  baselines_for_pairing: [B-hgb_all, B-mlp_pool]   # references for the GNN-vs-baseline paired deltas
  paired_gnn_runs: [G-gcn, G-sage, G-gat]          # only these runs are paired against the references (step 08 runtime)
  paired_runs:                                     # explicit pairings on identical rows
    - [B5-P, B5-U]
    - [BT-P, BT-U]
    - [A1a, G-sage]        # bridge node features removed
    - [A1b, G-sage]        # bridge features removed and bridge codes masked from the input
    - [B1, G-sage]         # directed precedence edges
    - [D1-shuffle, G-sage] # edge-shuffled null graph
    - [D1-rewire0, G-sage] # degree-preserving rewired null graph
```

- [ ] **Step 5: Docs**: in `docs/INTERFACES.md` step 08, replace "Paired deltas appear for every non-baseline run
  against each of `eval.baselines_for_pairing`" with the restriction to `eval.paired_gnn_runs` plus `eval.paired_runs`.

- [ ] **Step 6: Verify**: `uv run pytest -q` all green; `uv run ruff check .` clean.

- [ ] **Step 7: Commit**: `eval: pair only the main GNN runs against the baselines; add ablation pairings (A1a, A1b, B1, D1) vs G-sage`

---

### Task 2: Secondary target through the grid (`targets: all`)

**Files:**
- Modify: `src/bridgegnn/driver.py:30` (target expansion)
- Modify: `config/experiments.yaml` (G-gcn, G-sage, G-gat, B-prevalence, B-hgb_all, B-mlp_pool, B-bridge_lr → `targets: all`)
- Modify: `docs/INTERFACES.md` step 07 (`targets: primary | secondary | all | [list]`)
- Test: `tests/test_driver.py`

**Interfaces:**
- Produces: `driver.resolve_targets(spec: str | list[str], cfg: dict) -> list[str]`; `"primary"` → `cfg.task.primary_targets`,
  `"secondary"` → `cfg.task.secondary_targets`, `"all"` → primary + secondary, a list → itself; unknown string → `ValueError`.
  `expand` calls it.

- [ ] **Step 1: Write the failing test** (append to `tests/test_driver.py`; the `cfg` fixture is the real `config/p2.yaml`)

```python
def test_expand_targets_all_includes_secondary(cfg):
    grid = {"default_arch": "sage", "experiments": [
        {"id": "G-sage", "family": "gnn", "base": "gnn", "targets": "all", "overrides": {"arch": "sage"}},
        {"id": "B-lr", "family": "baseline", "base": "baseline", "targets": "primary", "overrides": {"model": "lr"}}]}
    jobs = driver.expand(grid, cfg, None)
    assert [j["target"] for j in jobs if j["run_id"] == "G-sage"] == cfg["task"]["primary_targets"] + cfg["task"]["secondary_targets"]
    assert [j["target"] for j in jobs if j["run_id"] == "B-lr"] == cfg["task"]["primary_targets"]
    with pytest.raises(ValueError):
        driver.resolve_targets("tertiary", cfg)

def test_same_fraction_and_seed_give_identical_subsets():
    ids = pl.Series("person_id", list(range(1, 401)) * 3)
    assert driver.subsample_patients(ids, 0.25, 3) == driver.subsample_patients(ids, 0.25, 3)
    assert driver.subsample_patients(ids, 0.25, 3) != driver.subsample_patients(ids, 0.25, 4)
```

- [ ] **Step 2: Run**: `uv run pytest tests/test_driver.py::test_expand_targets_all_includes_secondary -v` → FAIL (`"all"` treated as a list of characters, or AttributeError).

- [ ] **Step 3: Implement `resolve_targets`** in `driver.py`; replace the inline ternary in `expand` with it.

- [ ] **Step 4: Grid**: set `targets: all` on the seven entries named above. Leave every other entry `primary`.

- [ ] **Step 5: Verify dry run**: `uv run python scripts/07_experiments.py --dry-run | grep -c cancer_metastatic` → `7`
  (3 GNN + 4 baselines); `--dry-run --only-baselines | grep -c cancer_metastatic` → `4`.

- [ ] **Step 6: Smoke the real target once**: `uv run python scripts/06_gnn.py --run-id SMOKE-meta --target cancer_metastatic --arch sage --quick --seeds 0`
  exits 0 and prints one JSON line; then delete its outputs: `rm outputs/predictions/SMOKE-meta_* outputs/logs/06_SMOKE-meta_*` and
  remove the `SMOKE-meta` rows from `outputs/results/gnn.csv` (polars filter, rewrite). Confirms the bridge columns
  `bridge_flag_cancer_metastatic` / `bridge_score_cancer_metastatic` exist in the node matrix and the masked column resolves.

- [ ] **Step 7: Docs**: INTERFACES step 07 entry grammar; one line in `docs/DESIGN.md` open questions: metastatic kept as a
  secondary target with the same protocol, reported in the supplement (closes open question 2).

- [ ] **Step 8: Verify**: `uv run pytest -q`; `uv run ruff check .`.

- [ ] **Step 9: Commit**: `grid: run the main GNNs and four baselines on the secondary target (targets: all)`

---

### Task 3: One-command reproduction `scripts/run_all.sh` and artifact collection

**Files:**
- Create: `scripts/run_all.sh`
- Create: `src/bridgegnn/artifacts.py`, `scripts/collect_artifacts.py`
- Modify: `outputs/artifacts/README.md` (drop the stale `baseline_eval/` row; add `REPRODUCTION.md`, `pre2020/`, metastatic note)
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `artifacts.collect(out: Path, dest: Path) -> list[Path]`: copies the aggregate set into `dest`, returns the
  copied paths, refuses (ValueError) any source whose suffix is `.parquet` or `.txt`. The set:
  `results/*.csv`, `eval/tables/*.{csv,tex}`, `eval/figures/*.pdf`, `layers/*.csv`, `motifs/*.{csv,pdf}`,
  `pre2020/{summary,bridges,target_definitions,node_features}.csv`, `features_xcheck/{node_index.csv,node_x.npy,feature_groups.json}`,
  `features/{nodes.csv,graph_edges.csv,node_index.csv,node_x.npy,feature_groups.json,construction_thresholds.json}` (top level of `dest`, as today).
- `scripts/collect_artifacts.py [--out outputs] [--dest outputs/artifacts]` calls it and prints the count.
- `scripts/run_all.sh [--from NN] [--check]`: runs, in order, `00 01 02 03 04 05 06 07 09 08 08b collect`, where
  `05` = `07_experiments.py --only-baselines`, `06` = `07_experiments.py --ids G-gcn G-sage G-gat`, `07` = `07_experiments.py` with
  the remaining GNN ids (every `base: gnn` id except G-*), `08b` = `08_report_baselines.py`, `collect` = `collect_artifacts.py`.
  `--from NN` starts at that label. `--check` runs only `uv run pytest -q`, `07_experiments.py --dry-run`, `00_check_inputs.py`.
  Each step is wrapped in `run_step LABEL cmd...`: prints and appends `START/END label, ISO timestamp, elapsed seconds, exit code`
  to `outputs/logs/run_all.log`; `set -euo pipefail`; first failure stops the script with the label in the message.

- [ ] **Step 1: Write the failing tests**

```python
def test_collect_copies_aggregates_and_refuses_patient_files(tmp_path):
    out = tmp_path / "outputs"
    for rel in ["results/gnn.csv", "eval/tables/runs_agg.csv", "eval/figures/roc.pdf", "layers/layer_stats.csv",
                "motifs/triad_census.csv", "pre2020/summary.csv", "features/nodes.csv", "features_xcheck/node_x.npy"]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True); (out / rel).write_bytes(b"x")
    (out / "rows.parquet").write_bytes(b"x"); (out / "predictions").mkdir(); (out / "predictions/a.parquet").write_bytes(b"x")
    copied = artifacts.collect(out, tmp_path / "dest")
    rel = sorted(str(p.relative_to(tmp_path / "dest")) for p in copied)
    assert rel == ["eval/figures/roc.pdf", "eval/tables/runs_agg.csv", "features_xcheck/node_x.npy", "layers/layer_stats.csv",
                   "motifs/triad_census.csv", "nodes.csv", "pre2020/summary.csv", "results/gnn.csv"]
    assert not list((tmp_path / "dest").rglob("*.parquet"))

def test_collect_refuses_parquet_in_the_set(tmp_path):
    with pytest.raises(ValueError):
        artifacts._check_safe(Path("results/x.parquet"))
```

- [ ] **Step 2: Run** → FAIL (`ModuleNotFoundError: bridgegnn.artifacts`).

- [ ] **Step 3: Implement** `artifacts.py` (`PATTERNS: list[tuple[str, str]]` of (glob relative to `out`, dest subdir or "")),
  `_check_safe(path)`, `collect`. Missing sources are skipped silently (a partial tree is legal mid-run). Then `scripts/collect_artifacts.py`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Write `scripts/run_all.sh`** per the interface above; `chmod +x`. The remaining-GNN id list is computed
  from the grid, not hard-coded: `uv run python -c "import yaml;g=yaml.safe_load(open('config/experiments.yaml'));print(' '.join(e['id'] for e in g['experiments'] if e['base']=='gnn' and not e['id'].startswith('G-')))"`.

- [ ] **Step 6: Check mode**: `./scripts/run_all.sh --check` exits 0; `outputs/logs/run_all.log` has START/END lines for `check`.

- [ ] **Step 7: Resume check** (Review Focus 5): `./scripts/run_all.sh --from collect` on the current outputs runs only
  `collect` (log shows a single START/END pair) and `git status` shows no change under `outputs/artifacts/` beyond files the
  collection set newly includes.

- [ ] **Step 8: Artifacts README**: table rows match `PATTERNS`; remove `baseline_eval/`; add a line that `REPRODUCTION.md`
  (Task 6) records the fresh-clone comparison.

- [ ] **Step 9: Verify**: `uv run pytest -q`; `uv run ruff check .`; `bash -n scripts/run_all.sh`.

- [ ] **Step 10: Commit**: `repro: run_all.sh chains steps 00-09 through the driver; collect_artifacts copies the aggregate set`

---

### Task 4: Run comparison `scripts/compare_runs.py`

**Files:**
- Create: `src/bridgegnn/compare.py`, `scripts/compare_runs.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces: `compare.compare_dirs(old: Path, new: Path, tol: float = 0.02) -> pl.DataFrame` with columns
  `file, column, status, max_abs_diff, n_rows_old, n_rows_new, note`; `status ∈ {exact, within_tol, exceeds, missing_file, missing_column}`.
  Files compared: every `*.csv` under `results/`, `eval/tables/`, `layers/`, `motifs/`, `pre2020/` of both trees, plus
  `node_x.npy` and `features_xcheck/node_x.npy` (one row, column `matrix`, `max_abs_diff` = `np.abs(a-b).max()`).
  Alignment: rows joined on all non-numeric columns of the file (sorted); unmatched rows → `note="rows differ"` and
  status `exceeds`. Numeric columns: exact when `max_abs_diff == 0`; otherwise `within_tol` only if the file is
  `results/gnn.csv` or `eval/tables/{runs,runs_agg,runs_agg_rounded,paired,reliability,strata}.csv`, the row's `model`
  (or `run_id` prefix) is not a deterministic tabular model (`prevalence, lr, hgb, hgb_all, bridge_lr`) and
  `max_abs_diff <= tol`; else `exceeds`. String columns (incl. `params_json`): any inequality → `exceeds`.
- `compare.to_markdown(df: pl.DataFrame) -> str` and `compare.summary(df) -> dict` (`counts per status`).
- CLI: `uv run python scripts/compare_runs.py OLD_ARTIFACTS NEW_ARTIFACTS [--tol 0.02] [--md out.md]`; exit 1 if any `exceeds` or `missing_*`.

- [ ] **Step 1: Write the failing tests** (`tests/test_compare.py`; helper `write(dir, rel, df)`)

```python
def test_compare_identical_trees_are_exact(tmp_path): ...  # two copies of a results/gnn.csv with model sage → every status exact

def test_compare_within_tolerance_for_torch_rows_only(tmp_path):
    # results/gnn.csv: row (G-sage, cardiometabolic, 0, model sage) auroc 0.61 vs 0.62 → within_tol
    # results/baselines.csv: row (B-lr, ..., model lr) auroc 0.61 vs 0.62 → exceeds

def test_compare_params_json_must_match_exactly(tmp_path):
    # metrics equal, params_json '{"lr": 0.001}' vs '{"lr": 0.003}' → exceeds on column params_json

def test_compare_reports_missing_file_and_column(tmp_path):
    # new tree lacks motifs/triad_census.csv → missing_file; new runs_agg.csv lacks brier_mean → missing_column

def test_compare_npy_matrix(tmp_path):
    # node_x.npy equal → exact; perturbed by 1e-3 → exceeds (matrices are deterministic)

def test_cli_exit_code(tmp_path):  # subprocess: identical → 0; exceeds → 1, and --md writes the table
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement `compare.py`** (polars join on key columns; `TORCH_FILES`, `DETERMINISTIC_MODELS` constants) and the CLI.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Self-compare**: `uv run python scripts/compare_runs.py outputs/artifacts outputs/artifacts` → all exact, exit 0.

- [ ] **Step 6: Verify**: `uv run pytest -q`; `uv run ruff check .`.

- [ ] **Step 7: Commit**: `repro: compare_runs reports exact / within-tolerance / exceeding differences between two artifact trees`

---

### Task 5: README "Reproduce from scratch" and interface docs

**Files:**
- Modify: `README.md` (replace "Reproduce the tabular comparison" with a full section; keep the tabular subsection as a sub-heading)
- Modify: `docs/INTERFACES.md` (append `run_all.sh`, `collect_artifacts.py`, `compare_runs.py` entries)

- [ ] **Step 1: Write the README section** with exactly this flow:

```bash
git clone https://github.com/Nithin2311/project2-bridge-gnn.git && cd project2-bridge-gnn
# Project 1 must sit at ../project1-bridge-dx with its outputs/ (frozen, 2026-10-23); EHRShot CSVs at $EHRSHOT_DATA
uv sync --extra dev --extra gnn
export EHRSHOT_DATA="/path/to/FinalProject/data"
./scripts/run_all.sh --check            # tests, dry-run of the grid, input check (~2 min)
./scripts/run_all.sh                    # ~9 h on an M4 (grid ~6 h, baselines ~1.5 h, evaluation <15 min)
uv run python scripts/compare_runs.py /path/to/reference/outputs/artifacts outputs/artifacts --md REPRODUCTION.md
```

  Then: expected runtimes per step (from the design), what `outputs/artifacts/` contains, what "exact" and
  "within tolerance" mean, and the pointer to `outputs/artifacts/REPRODUCTION.md`.

- [ ] **Step 2: Verify** the README commands match the script's flags (`grep -- "--check\|--from" scripts/run_all.sh`).

- [ ] **Step 3: Commit**: `docs: reproduce-from-scratch instructions; interfaces for run_all, collect_artifacts, compare_runs`

---

### Task 6: Push, fresh clone, launch the re-run, record the commit

No code. Shell steps, in order; stop at the first failure and report.

- [ ] **Step 1**: `uv run pytest -q` green, `uv run ruff check .` clean, `git status` clean on `main`.
- [ ] **Step 2**: `git push origin main`. If the push is blocked, continue with Step 3 from the local repo (`git clone "$PWD" ...`) and note it.
- [ ] **Step 3**: `cd .. && git clone https://github.com/Nithin2311/project2-bridge-gnn.git project2-bridge-gnn-rerun && cd project2-bridge-gnn-rerun && git rev-parse HEAD` → record the hash.
- [ ] **Step 4**: `uv sync --extra dev --extra gnn && uv run pytest -q` → 150+ passed.
- [ ] **Step 5**: `./scripts/run_all.sh --check` → exit 0.
- [ ] **Step 6**: `nohup ./scripts/run_all.sh > outputs/logs/run_all.stdout 2>&1 &` ; record the PID and the start time.
- [ ] **Step 7**: After ~10 minutes confirm `outputs/logs/run_all.log` shows `END 01` with exit 0 and `START 02`.
- [ ] **Step 8**: Tell the user: commit hash, PID, log path, expected finish time; remind them to delete the remote branch
  `dylan/project2-deliverables` on GitHub (L5).

---

### Task 7 (after the re-run finishes, ~Nov 24): compare, record, replace artifacts on `main`

- [ ] **Step 1**: `cd project2-bridge-gnn-rerun && uv run python scripts/compare_runs.py ../project2-bridge-gnn/outputs/artifacts outputs/artifacts --md /tmp/compare.md`; note the exit code.
- [ ] **Step 2**: Write `outputs/artifacts/REPRODUCTION.md` in the main checkout: header (reference commit, rerun commit, machine,
  start/end from `run_all.log`, per-step elapsed), the comparison table, and one paragraph per `exceeds` row explaining it
  (or "none"). Any `exceeds` that is not machine noise is a finding for the review panel.
- [ ] **Step 3**: Copy the rerun's `outputs/artifacts/` over the main checkout's (`rsync -a --delete --exclude REPRODUCTION.md`),
  `git diff --stat outputs/artifacts` for a sanity glance (no parquet, no txt), commit:
  `results: artifacts from the fresh-clone re-run (commit <hash>); REPRODUCTION.md records the comparison`, push.
- [ ] **Step 4**: The rerun clone stays on disk until submission (its `outputs/predictions/` are the archived predictions behind the tables).
