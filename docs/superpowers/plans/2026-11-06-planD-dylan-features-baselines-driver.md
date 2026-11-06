# Plan D (Dylan): step 02 features, step 05 non-graph baselines, step 07 experiment driver

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For Dylan:** this plan is written so you can start today against the synthetic fixtures (`tests/conftest.py`, `synth.make`) and never wait on Nithin. When `outputs/rows.parquet` and `outputs/row_codes.parquet` land (Plan 1, Task 5), the same code runs on real data. Work on branches `dylan/02-features`, `dylan/05-baselines`, `dylan/07-driver`; open a PR per task. Nithin will independently build 02 and 05 on `nithin/xcheck-*` branches and the two are compared row-for-row before either is trusted.

**Goal:** Produce the feature matrices every model shares (step 02), the tuned non-graph baselines the GNNs must beat (step 05), and the driver that runs the spec's experiment grid (step 07).

**Architecture:** Pure functions in `src/bridgegnn/features.py` and `src/bridgegnn/tabular.py` operate on contract-validated polars frames; `scripts/02_features.py`, `scripts/05_baselines.py`, `scripts/07_experiments.py` are thin CLIs. Model selection is 5-fold GroupKFold by `person_id` on development rows (train + val), refit on all development rows, one prediction pass on test. All outputs use the `predictions` and `results` contract schemas so step 08 (Nithin) can evaluate every model identically.

**Tech Stack:** polars, numpy, scikit-learn (LogisticRegression, HistGradientBoostingClassifier, MLPClassifier, GroupKFold, StandardScaler), pyyaml, pytest. No torch.

**Spec:** `docs/superpowers/specs/2026-11-06-project2-design.md` sections 5, 7, 8, 10. Contract: `docs/DATA-CONTRACT.md`. CLI contract: `docs/INTERFACES.md`.

## Global Constraints

- Test rows (`split == "test"`) never enter `fit`, `fit_transform`, scaler fitting, vocabulary selection, or model selection. Every function that fits takes development rows only; tests assert a `ContractError`/`AssertionError` if a test row sneaks in.
- Seeds: `cfg.seeds = [0, 1, 2, 3, 4]`; every stochastic call receives the seed; results are reproducible run to run.
- Feature-group prefixes are fixed: `c_` (CODE bag, 496 main-graph nodes, masked per target), `all_` (all-code bag), `demo_`, `util_`, `temp_`, `bridge_`, `pool_mean_`, `pool_max_`. Group membership is written once to `outputs/features/feature_groups.json` and read by everyone.
- One `tabular_{target}_{split}.parquet` per (target, split); masking and bridge features are target-specific, so bags differ per target.
- `results.csv` rows and `predictions` parquet files follow `contract.SCHEMAS`; `run_id` for baselines is `B-<model>`.

## Review Focus

1. A code masked for the target must be absent from `c_` and `all_` columns and from the pooled features for that target's tables, but present for other targets. Test in Task 1.
2. An `all_` column for a category seen in development rows but never in test rows must still exist in the test table (schema identical across splits, zero-filled). Test in Task 1.
3. A target with zero positives in one CV fold (possible for cardiometabolic, 107 development positives over 5 folds) must not crash AUPRC; use `average_precision_score` and skip-and-warn folds with a single class. Test in Task 2.
4. `--subsample 0.25` must subsample development patients, not rows, with the seed, and leave the test set untouched. Test in Task 3.
5. The driver must `--dry-run` cleanly when `scripts/06_gnn.py` does not exist yet (prints commands, exit 0). Test in Task 3.

---

### Task 1: Step 02, feature matrices

**Files:**
- Create: `src/bridgegnn/features.py`, `scripts/02_features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `outputs/rows.parquet`, `outputs/row_codes.parquet` (contract `rows`, `row_codes`); Project 1 `outputs/p2/node_features.csv`, `outputs/bridges.csv`, `outputs/nodes.csv`; `cfg` from `bridgegnn.config.load()`.
- Produces:
  - `features.node_table(node_features: pl.DataFrame, bridges: pl.DataFrame, targets: list[str]) -> pl.DataFrame`: one row per main-graph category with columns `category`, the STRUCT columns (`prevalence, degree, weighted_degree, betweenness, bt_null_percentile, closeness, eigenvector, pagerank, participation, within_module_z, degree_residual_z, inpatient_share, ccir_chronic_share, mean_age_first_occurrence, female_share, backbone_degree`, `chapter_<letter>` one-hots, `community_<id>` one-hots for seed-0 communities with >= 5 nodes else `community_other`), and the BRIDGE columns `bridge_flag` (from `bridge_survivor`), `bridge_flag_<target>` (survivor on any pair containing the target; pairs are `"<a>-<b>"` strings in `bridges.csv`), `bridge_score_<target>` (max of `bridge_betweenness` over pairs containing the target; 0 if none).
  - `features.fit_node_scaler(node_table, cols) -> dict[str, tuple[float, float]]` (mean, sd per column) and `features.apply_node_scaler(node_table, scaler) -> pl.DataFrame`. The node table has no split (it is a train+val artifact), so scaling here is over all 496 nodes.
  - `features.write_node_artifacts(node_table_scaled, out_dir) -> None` writing `node_index.csv` (contract `node_index`), `node_x.npy` (float32, rows in `node_index` order), `feature_groups.json` mapping `STRUCT`, `BRIDGE`, `CHAPTER`, `COMMUNITY` to column lists plus the row-level groups `CODE`, `ALL`, `DEMO`, `UTIL`, `TEMP`, `POOL`.
  - `features.tabular_for_target(rows, row_codes, node_table_scaled, target: str, cfg, vocab_all: list[str] | None, lookback: int = 1095) -> tuple[pl.DataFrame, list[str]]`: returns the table (contract `tabular`) for the given target over all splits and the `vocab_all` used. Columns: `c_<cat>` for every main-graph category in `node_index` order, 1.0 if present in the lookback window and not masked for the target else 0.0; `all_<cat>` for `vocab_all` (when `None`, build it as every category with >= 10 development rows in the window, excluding masked codes for this target, sorted); `demo_age`, `demo_female`; `util_n_codes`, `util_n_rows`, `util_n_visit_days`, `util_days_since_first_dx`, `util_index_year`; `temp_min_days_since_last`, `temp_mean_days_since_last`, `temp_share_recent` (share of present codes with `days_since_last <= cfg.windows.recent_days`); `bridge_n_flag` (count of present unmasked `bridge_flag_<target>` codes), `bridge_sum_score`, `bridge_max_score` (over present unmasked codes, `bridge_score_<target>`); `pool_mean_<f>` and `pool_max_<f>` for every STRUCT + BRIDGE column `f` over present unmasked main-graph codes (0.0 when no codes).
  - `features.fit_row_scaler(table, cols, dev_mask) -> dict` / `features.apply_row_scaler(table, scaler)` for `demo_`, `util_`, `temp_`, `bridge_`, `pool_` columns; fit on development rows only; asserts no test row in `dev_mask`.
  - `features.write_graph_artifacts(p1_outputs: Path, p1_cfg: dict, out_dir: Path) -> None`: copies the aggregate deliverables the rubric asks for into `outputs/artifacts/`: `nodes.csv`, `graph_edges.csv` (from Project 1), `node_index.csv`, and writes `construction_thresholds.json` holding the Project 1 edge rule (`fisher_exact`, `bonferroni`, `alpha 0.05`, `min_odds_ratio 1.2`, `min_shared_patients 10`, `weight phi`), node rule (`min_prevalence 0.01`, excluded chapters R, Z, O), and the Project 2 window and masking settings from `p2.yaml`. These files are aggregate (category level) and are the only CSVs whitelisted for commit.
- `scripts/02_features.py` runs the above for every target in primary + secondary, writes `outputs/features/tabular_{target}_{split}.parquet` via `contract.write`, writes the node and graph artifacts, and prints per target: table shape, number of `all_` columns, positive counts per split, and the share of rows with `util_n_codes == 0` after masking.

- [ ] **Step 1: Write failing tests** `tests/test_features.py` (fixtures: `synth`, plus `rows_rc` built by Plan 1's `rows.build_*` on the synthetic frames; a `node_features` frame synthesised from `synth.nodes` with the STRUCT columns filled with random values and `community_seed0` in {0,1,2}):

```python
def test_node_table_has_bridge_columns_per_target(node_feats, synth):
    nt = features.node_table(node_feats, synth.bridges, TARGETS)
    for t in TARGETS:
        assert f"bridge_flag_{t}" in nt.columns and f"bridge_score_{t}" in nt.columns
    assert nt.height == synth.nodes.height

def test_bridge_score_is_max_over_pairs(node_feats, synth):
    ... pick a category with two pairs containing "mental_health"; assert equals max bridge_betweenness

def test_masked_code_absent_from_bag_for_that_target_only(rows_rc, nt, cfg):
    tab_cm, _ = features.tabular_for_target(rows, rc, nt, "cardiometabolic", cfg, None)
    tab_mh, _ = features.tabular_for_target(rows, rc, nt, "mental_health", cfg, None)
    cat = next(iter(masks["cardiometabolic"]))   # a main-graph code in the CM mask
    assert (tab_cm[f"c_{cat}"] == 0).all()
    assert (tab_mh[f"c_{cat}"] > 0).any()

def test_all_bag_schema_identical_across_splits(...):
    tab, vocab = ...; cols = {s: set(tab.filter(pl.col("split") == s).columns) for s in ["train", "val", "test"]}
    assert cols["train"] == cols["test"]

def test_all_vocab_built_from_dev_rows_only(...):
    ... a category appearing only in test rows must not be in vocab

def test_pooled_features_zero_when_no_codes(...):
    ... row whose codes are all masked -> pool_mean_* == 0 and util_n_codes == 0

def test_row_scaler_rejects_test_rows(...):
    with pytest.raises(AssertionError): features.fit_row_scaler(tab, cols, dev_mask=tab["split"] != "nothing")

def test_tabular_matches_contract(...):
    contract.validate(tab, "tabular")

def test_node_artifacts_roundtrip(tmp_path, nt):
    features.write_node_artifacts(nt, tmp_path); x = np.load(tmp_path / "node_x.npy")
    idx = pl.read_csv(tmp_path / "node_index.csv"); groups = json.load(open(tmp_path / "feature_groups.json"))
    assert x.shape == (idx.height, sum(len(groups[g]) for g in ["STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY"]))
    assert x.dtype == np.float32
```

- [ ] **Step 2: Run** `uv run pytest tests/test_features.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `features.py`** (polars pivots for bags: `row_codes.filter(window & ~masked).pivot(...)`; keep everything vectorised).

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Implement `scripts/02_features.py`** (loop targets, write, print).

- [ ] **Step 6: Run on synthetic data end to end**: `uv run python scripts/02_features.py --synthetic` (add this flag: when set, build inputs from `synth.make(0)` + Plan 1's `rows` functions into a temp dir and run the same code path). Expected: tables written, no error.

- [ ] **Step 7: Commit, push, open PR** `dylan/02-features`.

---

### Task 2: Step 05, non-graph baselines

**Files:**
- Create: `src/bridgegnn/tabular.py`, `scripts/05_baselines.py`
- Test: `tests/test_tabular.py`

**Interfaces:**
- Consumes: `outputs/features/tabular_{target}_{split}.parquet`, `feature_groups.json`, `cfg`.
- Produces:
  - `tabular.MODELS = ["prevalence", "lr", "hgb", "hgb_all", "mlp", "mlp_pool", "bridge_lr"]` and `tabular.feature_columns(model: str, columns: list[str]) -> list[str]`:
    - `prevalence`: `[]`
    - `lr`: `c_*` + `demo_*` + `util_*`
    - `hgb`: `c_*` + `demo_*` + `util_*` + `bridge_*`
    - `hgb_all`: `all_*` + `demo_*` + `util_*`
    - `mlp`: same as `hgb`
    - `mlp_pool`: `c_*` + `demo_*` + `util_*` + `temp_*` + `bridge_*` + `pool_mean_*` + `pool_max_*`
    - `bridge_lr`: `bridge_*` + `demo_*` + `util_*`
  - `tabular.grid(model) -> list[dict]`: `lr`: `C in [0.01, 0.1, 1.0]`, `class_weight="balanced"`, `max_iter=2000`; `hgb`/`hgb_all`: `learning_rate in [0.03, 0.1]`, `max_leaf_nodes in [15, 31]`, `max_iter=300`, `early_stopping=False`, `class_weight="balanced"` (sklearn >= 1.2); `mlp`/`mlp_pool`: `hidden_layer_sizes in [(64,), (128, 64)]`, `alpha in [1e-4, 1e-3]`, `max_iter=300`, `early_stopping=True`; `bridge_lr`: as `lr`; `prevalence`: `[{}]`.
  - `tabular.make(model, params, seed) -> sklearn estimator` (Pipeline with `StandardScaler(with_mean=False)` for lr/mlp variants; none for hgb; `DummyClassifier(strategy="prior")` for prevalence).
  - `tabular.select(model, X_dev, y_dev, groups_dev, seed, n_folds=5) -> tuple[dict, float, pl.DataFrame]`: for each grid point, GroupKFold out-of-fold probabilities; score = mean fold `average_precision_score`, folds with one class skipped with a warning; returns best params, best score, and the OOF predictions frame for the best params.
  - `tabular.run(model, target, tables: dict[str, pl.DataFrame], cfg, seed) -> tuple[pl.DataFrame, dict]`: select on dev (train + val concatenated), refit on dev with best params, predict test; returns `predictions` frame (test rows plus OOF dev rows flagged by `split`) and a `results` dict with `run_id=f"B-{model}"`, `family="baseline"`, `layer="none"`, `features=",".join(group names used)`, `params_json`, `dev_oof_auprc`, `test_auroc`, `test_auprc`, `test_brier`, `n_test_rows`, `n_test_pos`.
- `scripts/05_baselines.py --target T --model M [--seeds 0 1 2 3 4] [--synthetic]` appends to `outputs/results/baselines.csv` and writes `outputs/predictions/B-{model}_{target}_seed{s}.parquet`; prints one JSON summary line per seed; logs grid scores to `outputs/logs/05_{target}_{model}.log`.

- [ ] **Step 1: Write failing tests** `tests/test_tabular.py` (fixture `tables` built from Task 1 on synthetic data):

```python
def test_feature_columns_per_model(tables):
    cols = tables["train"].columns
    assert all(c.startswith(("c_", "demo_", "util_")) for c in tabular.feature_columns("lr", cols))
    assert not any(c.startswith("c_") for c in tabular.feature_columns("hgb_all", cols))
    assert any(c.startswith("pool_mean_") for c in tabular.feature_columns("mlp_pool", cols))

def test_select_uses_group_folds(tables):
    ... monkeypatch GroupKFold.split to record groups; assert no person_id appears in both train and held-out of a fold

def test_select_handles_single_class_fold(monkeypatch):
    ... construct y with 2 positives in 60 rows so some folds have no positive; assert returns finite score and warns

def test_run_never_fits_on_test(tables, cfg):
    preds, res = tabular.run("lr", "cardiometabolic", tables, cfg, seed=0)
    contract.validate(preds, "predictions")
    assert res["n_test_rows"] == tables["test"].height
    ... monkeypatch estimator.fit to assert no test person_id in the fitted X's index

def test_run_is_reproducible(tables, cfg):
    p1, _ = tabular.run("mlp", "mental_health", tables, cfg, seed=1); p2, _ = ...
    assert np.allclose(p1["p"], p2["p"])

def test_prevalence_baseline_constant(tables, cfg):
    preds, res = tabular.run("prevalence", "mental_health", tables, cfg, 0)
    assert preds.filter(pl.col("split") == "test")["p"].n_unique() == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_tabular.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `tabular.py`**.

- [ ] **Step 4: Run** the tests. Expected: PASS.

- [ ] **Step 5: Implement `scripts/05_baselines.py`** and run `uv run python scripts/05_baselines.py --target mental_health --model lr --seeds 0 --synthetic`. Expected: one results row appended, one predictions file, a log file.

- [ ] **Step 6: Commit, push, open PR** `dylan/05-baselines`.

- [ ] **Step 7 (when Plan 1 Task 5 has landed): real-data run** for all seven models x three primary targets x five seeds:

```bash
for t in cardiometabolic cancer_hematologic mental_health; do for m in prevalence lr hgb hgb_all mlp mlp_pool bridge_lr; do
  uv run python scripts/05_baselines.py --target $t --model $m; done; done
```

Expected: 105 results rows; wall time under 2 hours on the M4. Post the `results/baselines.csv` summary (mean test AUPRC per model x target) in the PR.

---

### Task 3: Step 07, experiment driver

**Files:**
- Create: `src/bridgegnn/driver.py`, `scripts/07_experiments.py`
- Test: `tests/test_driver.py`

**Interfaces:**
- Consumes: `config/experiments.yaml` (Plan 1 Task 6), `config/p2.yaml`, the step 05 and step 06 CLIs described in `docs/INTERFACES.md`.
- Produces:
  - `driver.expand(experiments: dict, cfg: dict, ids: list[str] | None) -> list[dict]`: one job per (experiment, target, seed-set) with fields `run_id`, `script` (`scripts/05_baselines.py` or `scripts/06_gnn.py`), `argv` (list of strings built from `overrides` plus defaults from `cfg`), `targets: primary` expanded from `cfg.task.primary_targets`.
  - `driver.subsample_patients(person_ids: pl.Series, fraction: float, seed: int) -> list[int]`: deterministic patient-level subsample used by `--subsample` jobs (the driver writes the id list to `outputs/subsamples/{run_id}_seed{s}.txt` and passes `--subsample-file` to the script; `docs/INTERFACES.md` lists this flag).
  - `driver.run(jobs, dry_run: bool, log_dir: Path) -> list[int]`: runs each job with `subprocess.run`, captures stdout/stderr to `log_dir/{run_id}_{target}.log`, continues on failure, returns exit codes; with `dry_run` prints the commands and returns zeros, even if the script file does not exist.
- `scripts/07_experiments.py [--ids A1a B1 ...] [--dry-run] [--only-baselines]`.

- [ ] **Step 1: Write failing tests** `tests/test_driver.py`:

```python
def test_expand_resolves_primary_targets(cfg):
    jobs = driver.expand({"experiments": [{"id": "A1a", "family": "A", "base": "gnn", "targets": "primary", "overrides": {"features": "CODE,DEMO,UTIL,TEMP,STRUCT"}}]}, cfg, None)
    assert {j["argv"][j["argv"].index("--target") + 1] for j in jobs} == set(cfg["task"]["primary_targets"])
    assert all("--features" in j["argv"] for j in jobs)

def test_subsample_is_patient_level_and_seeded():
    ids = pl.Series(range(100)); a = driver.subsample_patients(ids, 0.25, 0); b = driver.subsample_patients(ids, 0.25, 0)
    assert a == b and len(a) == 25

def test_dry_run_without_script(tmp_path, cfg):
    jobs = [{"run_id": "X", "script": "scripts/does_not_exist.py", "argv": ["--target", "mental_health"], "target": "mental_health"}]
    assert driver.run(jobs, dry_run=True, log_dir=tmp_path) == [0]

def test_run_continues_after_failure(tmp_path):
    ... two jobs, first runs `python -c "import sys; sys.exit(3)"`, second exits 0 -> [3, 0] and two log files
```

- [ ] **Step 2: Run** `uv run pytest tests/test_driver.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `driver.py`** and `scripts/07_experiments.py`.

- [ ] **Step 4: Run** the tests (PASS) and `uv run python scripts/07_experiments.py --dry-run`. Expected: prints one command per (experiment, target); exit 0.

- [ ] **Step 5: Commit, push, open PR** `dylan/07-driver`.

- [ ] **Step 6 (week 3, after Plan 3 lands step 06):** `uv run python scripts/07_experiments.py --ids A1a A1b B1 B3 D1-rewire D1-shuffle E1 C1-25 C1-50`. Post the results summary in the PR.

---

### Task 4: Report sections (weeks 3-4, in the report repo)

Not code; listed so ownership is explicit. Dylan drafts: Related Works (at least 5 recent, directly relevant studies: start from the Project 1 bib plus Chet AAAI 2022, Singh BICOB 2026, Hancox 2024, Oss Boll 2024, GraphCare ICLR 2024, and the EHRShot paper; each with a sentence on similarity, difference, and gap); the baseline part of Model Details (models 1-5, grids, best configurations from `results/baselines.csv`); the baseline rows of the Results tables; the C1 and D1 paragraphs of Experimental Analysis. Numbers enter the LaTeX only as macros generated by step 08.
