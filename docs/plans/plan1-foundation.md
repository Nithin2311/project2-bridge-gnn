# Plan 1: Foundation, data contract, rows, Dylan package


**Goal:** Stand up the `project2-bridge-gnn` repository with its config, the frozen data contract, synthetic fixtures, step 00 (input check) and step 01 (label rows + per-row code sets), and publish the package Dylan needs to start steps 02/05/07 independently.

**Architecture:** A new `uv` project `bridgegnn` beside the frozen Project 1 repo, which it imports as a path dependency for the category map, DuckDB access and null models. Pure functions in `src/bridgegnn/` take polars DataFrames and are tested on synthetic fixtures; thin numbered scripts do the DuckDB I/O and call them. Every file crossing between people is described once in `contract.py` and validated on write and read.

**Tech Stack:** Python >= 3.11, uv, polars, duckdb, pyarrow, numpy, pyyaml, pytest; `bridgedx` (Project 1) as a path dependency. No torch in this plan.

**Spec:** `docs/DESIGN.md` (sections 3, 4, 5, 10 are the ones this plan implements).

## Global Constraints

- No patient-level file is ever committed: `.gitignore` excludes `*.parquet`, `*.csv` (except `outputs/artifacts/` whitelisted aggregate files added in later plans), `*.duckdb`, `outputs/*`, `.cache/`.
- Test patients (`split == "test"` in Project 1's `outputs/patient_split.csv`) never enter any statistic, scaler, or selection; functions that fit anything take a `dev_mask` and assert it excludes test rows.
- Primary lookback 1,095 days; others 365 and 0 (= all ICD-10-era history from 2015-10-01). Horizon 12 months. Target definition `core`. Censored rows excluded.
- Every parameter lives in `config/p2.yaml` with a comment giving its reason; scripts never hard-code a value the config holds.
- Project 1 outputs are read-only inputs at `../project1-bridge-dx/outputs/` (config `paths.p1_outputs`).
- Python dates are `datetime.date`; polars `pl.Date`; `person_id` is `pl.Int64`; categories are 3-character ICD-10 strings.

## Review Focus

1. A patient whose only codes in the window are masked for a target must still produce a row (with `n_codes_* = 0` after masking is applied downstream) and must not crash pooling. Test in Task 4.
2. A code recorded on the index date itself counts as before the prediction (inclusive `<= index`), consistent with Project 1 step 10. Test in Task 4.
3. A category present in `row_codes` but absent from `nodes.csv` (sub-floor or R/Z/O) must carry `in_main_graph = False` and be kept. Test in Task 4.
4. Two label rows for the same patient and index date but different targets must share one set of code rows (no duplication by target). Test in Task 4.
5. `p2_labels.parquet` from Project 1 contains both horizons and three definitions; filtering must leave exactly core / 12 / uncensored, and the counts must match `p2_label_counts.csv` (cardiometabolic 163 positives). Test in Task 3 against the real file (skipped when Project 1 outputs are absent).

---

### Task 1: Repository scaffold and config

**Files:**
- Create: `project2-bridge-gnn/pyproject.toml`, `.gitignore`, `README.md`, `config/p2.yaml`, `src/bridgegnn/__init__.py`, `src/bridgegnn/config.py`, `outputs/.gitkeep`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `bridgegnn.config.load(path=None) -> dict`; `bridgegnn.config.p1_outputs(cfg) -> Path`; `bridgegnn.config.outputs_dir(cfg) -> Path`; `bridgegnn.config.REPO_ROOT`.
- `pyproject.toml` declares `bridgedx = { path = "../project1-bridge-dx", editable = true }` under `[tool.uv.sources]` and `bridgedx` in dependencies, so `from bridgedx import cohort, data, null_models` works.

- [ ] **Step 1: Initialise the repo**

```bash
cd "/Users/nithinpalyam/Desktop/Academic Folders/Fall 2026/Network Analysis & ML with Graphs/project2-bridge-gnn"
git init -b main
```

- [ ] **Step 2: Write `pyproject.toml`** with name `bridgegnn`, version `0.1.0`, `requires-python = ">=3.11"`, dependencies: `duckdb>=1.1`, `polars>=1.9`, `pyarrow>=17`, `numpy>=2.0`, `pandas>=2.2`, `scipy>=1.14`, `scikit-learn>=1.5`, `networkx>=3.3`, `igraph>=0.11`, `pyyaml>=6.0`, `matplotlib>=3.9`, `tqdm>=4.66`, `bridgedx`; optional `dev = ["pytest>=8", "ruff>=0.6"]`, `gnn = ["torch>=2.4", "torch_geometric>=2.6"]`; hatchling build with `packages = ["src/bridgegnn"]`; `[tool.uv.sources] bridgedx = { path = "../project1-bridge-dx", editable = true }`; ruff line length 110.

- [ ] **Step 3: Write `.gitignore`**: copy Project 1's, replace the `report/` block with `outputs/*`, `!outputs/.gitkeep`, `!outputs/artifacts/`, `outputs/artifacts/*.parquet`, and add `checkpoints/`, `logs/*.log`.

- [ ] **Step 4: Write `config/p2.yaml`** with these keys and a reason comment on each (values from the spec):

```yaml
paths: {p1_repo: ../project1-bridge-dx, p1_outputs: ../project1-bridge-dx/outputs, outputs: outputs, data_env: EHRSHOT_DATA, data_default: ../FinalProject/data}
task: {definition: core, horizon_months: 12, exclude_censored: true, primary_targets: [cardiometabolic, cancer_hematologic, mental_health], secondary_targets: [cancer_metastatic]}
windows: {lookback_days: [365, 1095, 0], primary_lookback: 1095, era_start: "2015-10-01", recent_days: 90, visit_window_days: 365}
masking: {input_exclusion: [core, chapter_block]}   # eligibility stays core-only (spec 5)
split: {dev: [train, val], test: [test], cv_folds: 5, group_by: person_id}
seeds: [0, 1, 2, 3, 4]
bridge: {survivor_flag: bridge_survivor, score_source: bridges.csv, score_column: bridge_betweenness}
```

- [ ] **Step 5: Write the failing test** `tests/test_config.py`:

```python
def test_load_has_required_sections():
    cfg = config.load()
    assert cfg["task"]["horizon_months"] == 12
    assert cfg["windows"]["primary_lookback"] == 1095
    assert set(cfg["task"]["primary_targets"]) == {"cardiometabolic", "cancer_hematologic", "mental_health"}

def test_p1_outputs_path_is_absolute():
    assert config.p1_outputs(config.load()).is_absolute()
```

- [ ] **Step 6: Run** `uv sync --extra dev && uv run pytest tests/test_config.py -v`. Expected: FAIL (module missing).

- [ ] **Step 7: Implement `src/bridgegnn/config.py`** mirroring Project 1's `config.py`: `REPO_ROOT`, `load`, `p1_outputs(cfg) -> Path` (REPO_ROOT / paths.p1_outputs, resolved), `outputs_dir(cfg) -> Path` (mkdir), `data_dir(cfg)` (env then default).

- [ ] **Step 8: Run** `uv run pytest tests/test_config.py -v`. Expected: PASS. Also `uv run python -c "import bridgedx.cohort, bridgedx.data, bridgedx.null_models"`. Expected: no error.

- [ ] **Step 9: Write `README.md`** (layout, setup, the three scripts of this plan, the pointer to `docs/DATA-CONTRACT.md` and the two plans; copy the "Code only. No EHRShot data ever committed" line from Project 1).

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "chore: scaffold bridgegnn project with config and P1 path dependency"
```

- [ ] **Step 11: Create the private GitHub repo and push** (owner has approved a private repo under their account):

```bash
gh repo create Nithin2311/project2-bridge-gnn --private --source . --remote origin --push
```

Expected: remote `origin` set, `main` pushed. Then add Dylan as collaborator in the GitHub UI (his handle is not in this tree; ask him).

---

### Task 2: Data contract and synthetic fixtures

**Files:**
- Create: `src/bridgegnn/contract.py`, `src/bridgegnn/synth.py`, `tests/conftest.py`, `docs/DATA-CONTRACT.md`
- Test: `tests/test_contract.py`, `tests/test_synth.py`

**Interfaces:**
- Produces: `contract.SCHEMAS: dict[str, dict[str, pl.DataType]]` with keys `rows`, `row_codes`, `node_index`, `tabular`, `predictions`, `results`; `contract.validate(df: pl.DataFrame, name: str) -> pl.DataFrame` (raises `ContractError` listing missing columns and dtype mismatches; extra columns allowed only for `tabular`); `contract.write(df, path, name)` and `contract.read(path, name)` which validate.
- Produces: `synth.make(seed: int = 0, n_patients: int = 60, n_categories: int = 30) -> SynthData` dataclass with fields `labels` (schema of P1 `p2_labels.parquet`, all three definitions, both horizons, some censored), `codes_long` (person_id, category, condition_date), `visits` (person_id, visit_date), `demo` (person_id, sex, birth_date), `split` (person_id, split with train/val/test), `nodes` (category, name, chapter for 20 of the 30 categories), `target_defs` (schema of P1 `target_definitions.csv`, four targets), `bridges` (schema subset of P1 `bridges.csv`: pair, category, bridge_betweenness, bridge_pass), `graph` (networkx Graph on the 20 node categories, phi weights). Seeded, deterministic, with a planted precedence signal A -> B used by Plan 2.
- `tests/conftest.py` exposes fixture `synth` returning `synth.make(0)`.

Exact schemas (these are the contract; copy into `DATA-CONTRACT.md` as tables):

```
rows:        person_id Int64, index_date Date, target Utf8, split Utf8, positive Boolean, days_to_event Int64 (null if negative),
             age_at_index Float64, sex Utf8, index_year Int32, days_since_first_dx Int32,
             n_codes_lb365 Int32, n_codes_lb1095 Int32, n_codes_lball Int32, n_rows_lb1095 Int32, n_visit_days_365 Int32
row_codes:   person_id Int64, index_date Date, category Utf8, first_date Date, last_date Date,
             days_since_first Int32, days_since_last Int32, in_lb365 Boolean, in_lb1095 Boolean, in_main_graph Boolean,
             masked_cardiometabolic Boolean, masked_cancer_metastatic Boolean, masked_cancer_hematologic Boolean, masked_mental_health Boolean
node_index:  category Utf8, idx Int32            (496 rows in order; row i of node_x.npy is category idx i)
tabular:     person_id Int64, index_date Date, target Utf8, split Utf8, positive Boolean  + feature columns prefixed
             c_ (496 masked bag), all_ (all-code bag), demo_, util_, temp_, bridge_, pool_mean_, pool_max_   (extra columns allowed)
predictions: run_id Utf8, model Utf8, target Utf8, seed Int32, person_id Int64, index_date Date, split Utf8, y Boolean, p Float64
results:     run_id Utf8, family Utf8, model Utf8, target Utf8, seed Int32, layer Utf8, features Utf8, params_json Utf8,
             dev_oof_auprc Float64, test_auroc Float64, test_auprc Float64, test_brier Float64, n_test_rows Int32, n_test_pos Int32
```

- [ ] **Step 1: Write failing tests** `tests/test_contract.py`:

```python
def test_validate_accepts_matching_frame():
    df = pl.DataFrame({c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["rows"].items()})
    assert contract.validate(df, "rows").height == 0

def test_validate_rejects_missing_column():
    df = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64)})
    with pytest.raises(contract.ContractError, match="index_date"):
        contract.validate(df, "rows")

def test_validate_rejects_wrong_dtype():
    sch = contract.SCHEMAS["predictions"]
    df = pl.DataFrame({c: pl.Series([0], dtype=pl.Int64 if c != "p" else pl.Utf8) for c in sch})
    with pytest.raises(contract.ContractError, match="p"):
        contract.validate(df, "predictions")

def test_tabular_allows_extra_feature_columns():
    base = {c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["tabular"].items()}
    base["c_I10"] = pl.Series([], dtype=pl.Float64)
    assert contract.validate(pl.DataFrame(base), "tabular") is not None
```

and `tests/test_synth.py`:

```python
def test_synth_is_deterministic():
    a, b = synth.make(0), synth.make(0)
    assert a.labels.equals(b.labels) and a.codes_long.equals(b.codes_long)

def test_synth_has_all_splits_and_targets():
    s = synth.make(0)
    assert set(s.split["split"]) == {"train", "val", "test"}
    assert set(s.labels["target"]) == {"cardiometabolic", "cancer_metastatic", "cancer_hematologic", "mental_health"}
    assert s.labels.filter(pl.col("censored")).height > 0

def test_synth_nodes_subset_of_categories():
    s = synth.make(0)
    assert set(s.nodes["category"]) < set(s.codes_long["category"])
```

- [ ] **Step 2: Run** `uv run pytest tests/test_contract.py tests/test_synth.py -v`. Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `contract.py`** (`SCHEMAS`, `ContractError(ValueError)`, `validate`, `write`, `read`; `read` uses `pl.read_parquet` or `pl.read_csv` by suffix).

- [ ] **Step 4: Implement `synth.py`** with `make(...)`. Labels: build with the same rules as `bridgedx.labels.build_landmark_labels` by calling it on the synthetic frames (import it; this keeps synthetic labels structurally identical to real ones) using a small config dict with `grid_start 2017-10-01`, `grid_step_days 182`, `min_history_days 730`, `require_encounter_days 365`, `horizons_months [6, 12]`. Codes: each patient 5-25 categories over 2015-10 to 2023-06; plant `A -> B` precedence (B first date = A first date + 60..200 days) in 70% of patients who have both.

- [ ] **Step 5: Run** the two test files. Expected: PASS.

- [ ] **Step 6: Write `docs/DATA-CONTRACT.md`**: the schema tables above, the file names and locations (`outputs/rows.parquet`, `outputs/row_codes.parquet`, `outputs/features/node_index.csv`, `outputs/features/node_x.npy`, `outputs/features/feature_groups.json`, `outputs/features/tabular_{target}_{split}.parquet`, `outputs/predictions/{run_id}_{target}_seed{seed}.parquet`, `outputs/results/*.csv`), the feature-group names CODE / DEMO / UTIL / TEMP / STRUCT / BRIDGE with their column prefixes, and the rule "validate on write and read with `contract.write/read`".

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: data contract, synthetic fixtures, contract doc"
```

---

### Task 3: Step 00, input check

**Files:**
- Create: `scripts/00_check_inputs.py`, `src/bridgegnn/inputs.py`
- Test: `tests/test_inputs.py`

**Interfaces:**
- Produces: `inputs.P1_FILES = ["p2_labels.parquet", "p2_label_counts.csv", "patient_split.csv", "target_definitions.csv", "nodes.csv", "bridges.csv", "subgroup_labels.csv", "patient_category.parquet", "graph_main.graphml", "p2/node_features.csv", "p2/graph_backbone_topk.graphml"]`; `inputs.check(p1: Path) -> list[str]` returning missing files; `inputs.task_labels(labels: pl.DataFrame, cfg: dict) -> pl.DataFrame` filtering to `definition == cfg.task.definition`, `horizon_months == cfg.task.horizon_months`, `not censored` when `exclude_censored`, and dropping `definition`, `horizon_months`, `censored`; `inputs.class_balance(labels) -> pl.DataFrame` with columns target, split, rows, patients, positives, positive_rate.

- [ ] **Step 1: Write failing tests** `tests/test_inputs.py`:

```python
def test_task_labels_filters_to_core_12_uncensored(synth):
    cfg = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True}}
    out = inputs.task_labels(synth.labels, cfg)
    assert "censored" not in out.columns and "definition" not in out.columns
    raw = synth.labels.filter((pl.col("definition") == "core") & (pl.col("horizon_months") == 12) & ~pl.col("censored"))
    assert out.height == raw.height

def test_check_reports_missing(tmp_path):
    assert set(inputs.check(tmp_path)) == set(inputs.P1_FILES)

@pytest.mark.skipif(not REAL_P1.exists(), reason="Project 1 outputs not present")
def test_real_counts_match_label_counts():
    labels = inputs.task_labels(pl.read_parquet(REAL_P1 / "p2_labels.parquet"), config.load())
    bal = inputs.class_balance(labels)
    cm = bal.filter((pl.col("target") == "cardiometabolic") & (pl.col("split") == "all")).row(0, named=True)
    assert cm["positives"] == 163 and cm["rows"] == 1863 - 182
```

- [ ] **Step 2: Run** `uv run pytest tests/test_inputs.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `inputs.py`** (`class_balance` adds an `all` split row like P1's `labels.summarize`).

- [ ] **Step 4: Implement `scripts/00_check_inputs.py`**: load config; print missing P1 files (exit 1 if any); open DuckDB via `bridgedx.data.connect(p1_cfg)` where `p1_cfg = bridgedx.config.load(REPO_ROOT / paths.p1_repo / "config/option_a.yaml")` and print row counts of `condition_occurrence`, `visit_occurrence`, `person`, `death`; print the class-balance table for the task.

- [ ] **Step 5: Run** `uv run pytest tests/test_inputs.py -v` (PASS) and `uv run python scripts/00_check_inputs.py` (prints table; cardiometabolic positives 163).

- [ ] **Step 6: Commit** `git commit -am "feat: step 00 input check and task label filter"`.

---

### Task 4: Rows and row codes (pure functions)

**Files:**
- Create: `src/bridgegnn/rows.py`
- Test: `tests/test_rows.py`

**Interfaces:**
- Consumes: `inputs.task_labels` output; frames shaped like `synth.codes_long`, `synth.visits`, `synth.demo`, `synth.nodes`, `synth.target_defs`.
- Produces:
  - `rows.mask_sets(target_defs: pl.DataFrame, cfg: dict) -> dict[str, set[str]]`: per target, union of categories with `in_core` and, if `chapter_block` in `cfg.masking.input_exclusion`, `in_chapter_block`.
  - `rows.build_row_codes(index_rows: pl.DataFrame, codes_long: pl.DataFrame, nodes: pl.DataFrame, masks: dict[str, set[str]], cfg: dict) -> pl.DataFrame` where `index_rows` has unique (person_id, index_date); output matches `SCHEMAS["row_codes"]`; a (person, index, category) row exists iff the category has any recording with `condition_date <= index_date` and `>= era_start`; `in_lb365 = last_date > index_date - 365d`; `in_lb1095` likewise; `days_since_* = (index_date - *_date).days`.
  - `rows.build_rows(labels: pl.DataFrame, row_codes: pl.DataFrame, visits: pl.DataFrame, demo: pl.DataFrame, codes_long: pl.DataFrame, cfg: dict) -> pl.DataFrame` matching `SCHEMAS["rows"]`; `age_at_index = (index_date - birth_date).days / 365.25`; `n_codes_lb*` count main-graph categories in the window before masking (masking is applied per target downstream by Dylan's step 02 using `masked_<target>`); `n_rows_lb1095` = number of `codes_long` rows (not distinct) in the 1,095-day window; `n_visit_days_365` = distinct `visit_date` in `(index - 365d, index]`; `days_since_first_dx = (index_date - min era condition_date).days`; `index_year = index_date.year`.

- [ ] **Step 1: Write failing tests** `tests/test_rows.py` (use `synth` fixture; `labels = inputs.task_labels(synth.labels, cfg)`; `idx = labels.select("person_id", "index_date").unique()`):

```python
def test_row_codes_matches_contract(synth, cfg):
    rc = rows.build_row_codes(idx, synth.codes_long, synth.nodes, rows.mask_sets(synth.target_defs, cfg), cfg)
    contract.validate(rc, "row_codes")

def test_no_code_after_index(synth, cfg):
    rc = ...; assert (rc["days_since_last"] >= 0).all() and (rc["days_since_first"] >= rc["days_since_last"]).all()

def test_code_on_index_day_is_included(cfg):
    one = pl.DataFrame({"person_id": [1], "category": ["I10"], "condition_date": [date(2019, 1, 1)]})
    idx = pl.DataFrame({"person_id": [1], "index_date": [date(2019, 1, 1)]})
    rc = rows.build_row_codes(idx, one, nodes_with(["I10"]), {}, cfg)
    assert rc.height == 1 and rc["days_since_last"][0] == 0 and rc["in_lb365"][0]

def test_non_graph_category_kept_with_flag(cfg):
    ... category "Z79" not in nodes -> row present, in_main_graph False

def test_mask_sets_union_core_and_block(synth, cfg):
    m = rows.mask_sets(synth.target_defs, cfg)
    td = synth.target_defs.filter(pl.col("target") == "cardiometabolic")
    assert m["cardiometabolic"] == set(td.filter(pl.col("in_core") | pl.col("in_chapter_block"))["category"])

def test_row_codes_shared_across_targets(synth, cfg):
    rc = ...; assert rc.select("person_id", "index_date", "category").is_unique().all()

def test_rows_matches_contract_and_counts(synth, cfg):
    r = rows.build_rows(labels, rc, synth.visits, synth.demo, synth.codes_long, cfg)
    contract.validate(r, "rows"); assert r.height == labels.height
    assert (r["n_codes_lb365"] <= r["n_codes_lb1095"]).all() and (r["n_codes_lb1095"] <= r["n_codes_lball"]).all()

def test_row_with_only_masked_codes_survives(cfg):
    ... patient whose only code is in the cardiometabolic mask -> row exists with n_codes_lb1095 == 1 (masking is downstream)

def test_lookback_window_boundary(cfg):
    ... code dated index - 365 days exactly -> in_lb365 False; index - 364 -> True
```

- [ ] **Step 2: Run** `uv run pytest tests/test_rows.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `rows.py`** (polars joins and group_bys; no Python loops over patients).

- [ ] **Step 4: Run** `uv run pytest tests/test_rows.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: row and row-code builders with masking and window flags"`.

---

### Task 5: Step 01 script (DuckDB extraction on real data)

**Files:**
- Create: `scripts/01_rows.py`, `src/bridgegnn/extract.py`
- Test: `tests/test_extract.py` (SQL shape test on an in-memory DuckDB with tiny tables)

**Interfaces:**
- Produces: `extract.codes_long(con, person_ids: list[int], era_start: str) -> pl.DataFrame` (person_id, category, condition_date) for ALL categories (join `cat_map` registered by `bridgedx.cohort._register_category_map(con)`, drop rows after death, dates `>= era_start`, distinct per (person, category, date)); `extract.visits(con, person_ids) -> pl.DataFrame` (person_id, visit_date distinct, any visit_concept_id, after-death dropped); `extract.demo(con, person_ids) -> pl.DataFrame` (person_id, sex from `gender_concept_id` 8532 female / 8507 male, birth_date from `birth_DATETIME`).
- `scripts/01_rows.py` writes `outputs/rows.parquet` and `outputs/row_codes.parquet` via `contract.write`, prints per-target class balance and the lookback summary (median codes per window, share with zero main-graph codes), and refuses to run if any test patient id appears in `rows` with a split other than `test` (sanity against split drift).

- [ ] **Step 1: Write failing test** `tests/test_extract.py`: build an in-memory DuckDB with tables `condition_occurrence(person_id, condition_source_value, condition_start_DATE)`, `death(person_id, death_DATE)`, `visit_occurrence(person_id, visit_start_DATE, visit_concept_id)`, `person(person_id, gender_concept_id, birth_DATETIME)` and a stub `cat_map(sv, category)` view; assert a post-death row is dropped, a pre-era row is dropped, duplicates collapse, sex maps correctly.

- [ ] **Step 2: Run** it. Expected: FAIL.

- [ ] **Step 3: Implement `extract.py`** (three SQL strings; `cat_map` is expected to exist on `con`; the script registers it).

- [ ] **Step 4: Implement `scripts/01_rows.py`**: load both configs; `con = bridgedx.data.connect(p1_cfg)`; `_register_category_map(con)`; labels = `inputs.task_labels(read p2_labels)`; person_ids = labels' patients; extract; `rows.build_row_codes` with `nodes.csv` and `rows.mask_sets(target_definitions.csv)`; `rows.build_rows`; write; print.

- [ ] **Step 5: Run** `uv run pytest -q` (all PASS) then `uv run python scripts/01_rows.py`. Expected: prints cardiometabolic 163 positives; `outputs/rows.parquet` and `outputs/row_codes.parquet` exist; runtime under 5 minutes.

- [ ] **Step 6: Commit** `git commit -am "feat: step 01 rows and row codes from EHRShot via DuckDB"`.

---

### Task 6: Interface contract for step 06 and the Dylan package

**Files:**
- Create: `docs/INTERFACES.md`, `config/experiments.yaml`
- Create (outside the repo): `../for_dylan/README.md`, `../for_dylan/MESSAGE.md`, copies of `docs/DATA-CONTRACT.md`, `docs/INTERFACES.md`, the design, and the Dylan plan.

**Interfaces:**
- Produces `docs/INTERFACES.md` with:
  - The step 06 CLI that Dylan's step 07 driver calls (Plan 3 implements it; Plan D's driver must be able to dry-run before it exists):
    ```
    uv run python scripts/06_gnn.py --run-id A1a --target cardiometabolic --arch sage --layer U
        --features CODE,DEMO,UTIL,TEMP,STRUCT,BRIDGE --lookback 1095 --expand 0 --weighted 1
        --depth 2 --agg mean --heads 4 --hidden 64 --seeds 0 1 2 3 4 --subsample 1.0 --null-graph none
        [--drop-codes bridge] [--dry-run]
    ```
    Allowed values: `--layer U|U-bb|U-bh|D|N`; `--null-graph none|rewire0..rewire4|shuffle`; `--features` any subset of the six groups; `--subsample` in (0, 1]; `--drop-codes none|bridge`. Output: one `predictions` parquet per seed and one `results` row per seed appended to `outputs/results/gnn.csv`, using the contract schemas. Exit code 0 and a one-line JSON summary on stdout.
  - The step 05 CLI (Plan D): `scripts/05_baselines.py --run-id B-lr --target X --model prevalence|lr|hgb|hgb_all|mlp|mlp_pool|bridge_lr --seeds ...` with the same outputs.
  - `config/experiments.yaml`: one entry per experiment ID from the spec's section 8 table, each `{id, family, base: gnn|baseline, targets: primary, overrides: {...cli flags...}}`; the driver expands `targets: primary` from `p2.yaml`.
- `for_dylan/README.md`: what to install (`uv sync --extra dev` in the repo), what to read first (design sections 1, 2, 5, 7, 8, 10; DATA-CONTRACT; INTERFACES), his three steps with the Plan D link, how to run tests, the branch and PR convention (`dylan/<step>` branches, PR to `main`, Nithin reviews; Nithin's independent reimplementation of 02/05 lives on `nithin/xcheck-*` branches and is compared, not merged), what he must never commit, and the two dates: his synthetic-fixture version of steps 02 and 05 by Nov 12; real-data run by Nov 16.
- `for_dylan/MESSAGE.md`: the message Nithin pastes to Dylan (plain prose, normal tone): context from Dr. Kim's call, the ask, the two dates, the credit arrangement question, the GitHub handle request.

- [ ] **Step 1: Write `docs/INTERFACES.md`** and `config/experiments.yaml` (the experiment IDs A1a, A1b, A2, A3, A4, B1, B2, B3, B4, B5, B6, C1-25, C1-50, D1-rewire, D1-shuffle, D2, E1, E2-agg, E2-heads, E3 with their override flags).

- [ ] **Step 2: Write `for_dylan/README.md` and `for_dylan/MESSAGE.md`**; copy the four documents in.

- [ ] **Step 3: Verify** `ls ../for_dylan` shows 6 files; `uv run python -c "import yaml; yaml.safe_load(open('config/experiments.yaml'))"` passes.

- [ ] **Step 4: Commit and push**

```bash
git add docs/INTERFACES.md config/experiments.yaml && git commit -m "docs: step 05/06 CLI contract and experiment grid" && git push
```
