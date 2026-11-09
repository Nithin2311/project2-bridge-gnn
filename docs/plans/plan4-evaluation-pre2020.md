# Plan 4: evaluation (step 08), step 06 row and layer flags, pre-2020 rebuild and backtest (step 09)

**Goal:** Turn every predictions file into the rubric's numbers and figures (mean and SD, patient-bootstrap
intervals, calibration, F1 at an out-of-fold threshold, confusion matrices, stratified results, paired
GNN-vs-baseline differences, ROC/PR and training curves, LaTeX tables), and run experiment B5 plus the temporal
backtest on a graph rebuilt from development records before 2020-01-01.

**Architecture:** `eval/metrics.py` holds pure functions on numpy arrays (bootstrap, calibration, thresholds,
paired deltas); `scripts/08_evaluate.py` discovers `outputs/predictions/*.parquet`, joins `rows.parquet` for
strata, and writes `outputs/eval/{tables,figures}/`. Step 06 gains `--layer-file`, `--index-from`, `--index-to`
so one CLI serves B5 and the backtest. `rebuild.py` recomputes nodes (same 496 set), Fisher/phi edges, Leiden
cores, restricted-betweenness bridges and structural node features on development records dated before a
cutoff, using Project 1's own functions; `scripts/09_pre2020_graph.py` writes `outputs/pre2020/` and calls
step 06 for B5 (rows indexed >= 2020 on U vs P) and the backtest (train <= 2019-12-31, test >= 2020-07-01 on P).

**Tech Stack:** numpy, polars, scikit-learn (metrics, LogisticRegression for calibration slope), matplotlib,
networkx; `bridgedx.{edges, graph, communities, centrality, null_models}` for the rebuild.

**Spec:** `docs/DESIGN.md` sections 4 (backtest), 5a (B5), 8 (metrics, experiments); `docs/INTERFACES.md`.

## Global Constraints

- Bootstrap: 1,000 resamples of TEST PATIENTS (not rows), percentile 95% interval, seed 0; stated in every table
  caption as "patient-level percentile bootstrap".
- Calibration: Brier; slope and intercept from a logistic regression of `y` on `logit(p)` (p clipped to
  [1e-6, 1 - 1e-6]); reliability plot with 10 quantile bins.
- F1 threshold: chosen on development out-of-fold predictions (split != test) per (run_id, target, seed) to
  maximise F1; applied to test rows; confusion matrix at that threshold.
- Strata on test rows: `sex`, age band (< 65 / >= 65 at index), subgraph-size bin from `n_codes_lb1095`
  (< 3, 3-9, >= 10), `index_year`. Each stratum needs >= 10 positives to be reported, else "n/a".
- Paired comparison: for two run_ids on the same (target, seed), bootstrap the DIFFERENCE in AUROC and AUPRC
  over the same patient resamples; reported for every GNN run vs `B-hgb_all` and `B-mlp_pool` when those
  predictions exist (Dylan's step 05), else skipped with a note.
- Aggregation: per (run_id, target): mean and SD over seeds of test AUROC / AUPRC / Brier / F1, plus the
  bootstrap interval of the seed-0 refit (one interval per run; seeds are summarised by SD).
- Pre-2020 rebuild (step 09): development patients only; records with `condition_start_DATE < cutoff`
  (`rebuild.cutoff = "2020-01-01"`); node set fixed to the 496 P1 main-graph nodes (features align with
  `node_index`); edges by P1's Fisher + Bonferroni + OR + support rule on the restricted patient x category
  table (P1 `edges.pair_counts/fisher_filter/weights` with the P1 config); Leiden at P1 resolution, 100-seed
  co-assignment, stable cores with P1 anchors at 0.9; `cluster_bridges` with 100 nulls; structural node
  features recomputed (prevalence, degree, weighted_degree, betweenness, bt_null_percentile, closeness,
  eigenvector, pagerank, participation, within_module_z, degree_residual_z, backbone_degree); time-invariant
  columns (inpatient_share, ccir_chronic_share, mean_age_first_occurrence, female_share, community_seed0,
  bridge_survivor) copied from P1 with that stated. Written as `outputs/pre2020/graph.graphml`,
  `outputs/pre2020/node_features.csv`, `outputs/pre2020/bridges.csv`, `outputs/pre2020/features/` (nodefeat
  format), `outputs/pre2020/summary.csv` (edge count, Jaccard with U, ARI with P1 communities, bridge overlap).
- Config: `eval: {n_bootstrap: 1000, seed: 0, min_stratum_pos: 10, calib_bins: 10, baselines_for_pairing:
  [B-hgb_all, B-mlp_pool]}`, `rebuild: {cutoff: "2020-01-01", backtest_train_to: "2019-12-31",
  backtest_test_from: "2020-07-01"}`.

## Review Focus

1. A stratum with fewer than 10 positives must print "n/a", not a misleading AUROC. Test in Task 1.
2. The F1 threshold must come only from development out-of-fold rows; a test with no dev rows must raise,
   not fall back to test rows. Test in Task 1.
3. Paired bootstrap must resample the same patients for both runs; a test with two identical prediction sets
   must give a delta interval of exactly [0, 0]. Test in Task 1.
4. `--index-from` / `--index-to` filter BOTH development and test rows, and the backtest must leave no test row
   with index before `backtest_test_from`. Test in Task 3.
5. The rebuilt graph must contain only development patients' records before the cutoff: the rebuild function
   asserts no test patient and no date >= cutoff in its input table. Test in Task 4.

---

### Task 1: Metrics (`eval/metrics.py`)

**Files:** Create `src/bridgegnn/eval/__init__.py`, `src/bridgegnn/eval/metrics.py`; test `tests/test_eval_metrics.py`.

**Interfaces:**
- `metrics.bootstrap_ci(y, p, person_id, stat: callable, n: int, seed: int) -> tuple[float, float, float]`
  (point, lo, hi); resamples unique patients with replacement and takes all their rows.
- `metrics.auroc(y, p) -> float | None`, `metrics.auprc(y, p) -> float | None` (None if one class).
- `metrics.calibration(y, p) -> dict(brier, slope, intercept)` and `metrics.reliability(y, p, bins) -> pl.DataFrame(bin, p_mean, y_rate, n)`.
- `metrics.f1_threshold(y_dev, p_dev) -> float` (raises ValueError if `y_dev` is empty or single-class);
  `metrics.confusion(y, p, thr) -> dict(tp, fp, fn, tn, precision, recall, f1)`.
- `metrics.stratified(df: pl.DataFrame, strata: list[str], min_pos: int) -> pl.DataFrame` with columns
  `stratum, level, n, pos, auroc, auprc` (`auroc`/`auprc` null below `min_pos`), `df` having `y, p, person_id`
  and the stratum columns.
- `metrics.paired_delta(y, p_a, p_b, person_id, n, seed) -> dict(d_auroc, d_auroc_lo, d_auroc_hi, d_auprc, ...)`.

- [ ] Tests (write first, run RED, implement, GREEN, commit `feat: evaluation metrics`):
  - bootstrap of a perfect classifier gives (1, 1, 1); resampling is by patient (monkeypatch rng, assert all rows
    of a drawn patient appear together).
  - calibration slope ~1 and intercept ~0 for `p` drawn from the true probabilities (simulate 5,000 rows).
  - `f1_threshold` raises on empty dev; `confusion` sums to n.
  - stratified: a level with 5 positives has null auroc; totals per stratum equal n.
  - paired_delta identical inputs -> (0, 0, 0).

### Task 2: Step 08 script

**Files:** Create `scripts/08_evaluate.py`, `src/bridgegnn/eval/report.py` (table builders), `config/p2.yaml` `eval:`; test `tests/test_eval_report.py` (synthetic predictions written through `contract.write`).

**Interfaces:**
- `report.load_predictions(dir) -> pl.DataFrame` (all files, validated).
- `report.summarise(preds, rows, cfg) -> dict[str, pl.DataFrame]` with keys `runs` (per run_id x target x seed
  metrics), `runs_agg` (mean, sd, CI), `calibration`, `strata`, `confusion`, `paired`.
- Script writes `outputs/eval/tables/*.csv` and `*.tex` (booktabs, 3 decimals), figures
  `outputs/eval/figures/{roc,pr}_{target}.pdf` (main G-* runs, seed 0), `reliability_{target}.pdf`,
  `curves_{run_id}_{target}.pdf` (train/val loss and val AUPRC from the step 06 JSON), `strata_{target}.pdf`.
- [ ] Tests: `summarise` on two synthetic runs produces the six tables with expected columns and the paired table
  only when a baseline run_id exists. Commit `feat: step 08 evaluation tables and figures`.
- [ ] Run on the real outputs: `uv run python scripts/08_evaluate.py`; confirm tables for the 9 G-* runs.

### Task 3: Step 06 flags for B5 and the backtest

**Files:** Modify `scripts/06_gnn.py`; test `tests/test_cli_filters.py` (pure helper tested without torch).
- Add `--layer-file PATH` (graphml; overrides `--layer`, recorded as `layer: file:<name>`), `--index-from
  YYYY-MM-DD`, `--index-to YYYY-MM-DD` (inclusive; filter applied to `rows_t` before building examples),
  `--test-index-from` (filter test rows only; the backtest's test window). Extract
  `filter_rows(rows_t, index_from, index_to, test_index_from) -> pl.DataFrame` into `bridgegnn.gnn.data`.
- [ ] Tests: dev and test rows both filtered; `test_index_from` leaves no test row before it and leaves dev
  rows alone. Commit `feat: step 06 row-window and layer-file flags`.

### Task 4: Pre-2020 rebuild (`rebuild.py`, step 09)

**Files:** Create `src/bridgegnn/rebuild.py`, `scripts/09_pre2020_graph.py`, config `rebuild:`; test `tests/test_rebuild.py` (synthetic).

**Interfaces:**
- `rebuild.restricted_patient_category(con, cfg, p1_cfg, dev_ids, cutoff) -> pl.DataFrame` (person_id, category,
  first_date) from `condition_occurrence` rows `>= era_start` and `< cutoff`, post-death dropped, via `cat_map`;
  asserts the constraints in Review Focus 5.
- `rebuild.build_graph(pc, nodes_496: pl.DataFrame, p1_cfg) -> tuple[nx.Graph, pl.DataFrame]` using
  `bridgedx.edges.pair_counts(pc, categories)`, `fisher_filter`, `weights`, `bridgedx.graph.build` (prevalence
  recomputed on the restricted table and written into the node attrs).
- `rebuild.cores_and_bridges(g, p1_cfg) -> tuple[dict, pl.DataFrame, pl.DataFrame]`: partition (Leiden seed 0),
  `target_definitions`-like frame (core per target from 100-seed co-assignment), `bridges` frame from
  `bridgedx.centrality.cluster_bridges`.
- `rebuild.node_features(g, partition, bridges, p1_node_features) -> pl.DataFrame` with the columns
  `nodefeat.build` needs (recomputed structural columns; copied time-invariant columns; `bridge_survivor` from
  the new `bridges.bridge_pass`).
- `rebuild.compare(g_new, g_u, part_new, part_u, bridges_new, bridges_u) -> dict` (edge Jaccard, ARI, bridge overlap).
- Script 09: build and write `outputs/pre2020/*`; then `subprocess` step 06 four times per primary target:
  `B5-U` (layer U, `--index-from 2020-01-01`), `B5-P` (`--layer-file outputs/pre2020/graph.graphml
  --node-features-dir outputs/pre2020/features --index-from 2020-01-01`), `BT-P` (backtest: `--layer-file ...
  --index-to 2019-12-31 --test-index-from 2020-07-01`), `BT-U` (same windows on layer U, for reference); arch
  from `experiments.yaml` `default_arch`. Print the comparison summary.
- [ ] Tests: restricted table rejects a test id and a post-cutoff date; `build_graph` on synth with cutoff yields
  fewer edges than without; `node_features` has every `nodefeat.STRUCT_COLS` column. Commit `feat: step 09
  pre-2020 rebuild and backtest`.
- [ ] Run step 09 on real data (background; cluster_bridges with 100 nulls is the slow part, expect 10-20 min);
  record the summary (edges, Jaccard, ARI, bridge overlap) and the four runs' results in the ledger.

### Task 5: Docs and experiments grid

- [ ] `config/experiments.yaml`: add `B5-U`, `B5-P`, `BT-U`, `BT-P` entries (base: gnn, family B / backtest)
  for the record; `docs/INTERFACES.md` (new flags), `docs/DATA-CONTRACT.md` (`outputs/eval/`, `outputs/pre2020/`),
  README (steps 08, 09). Commit `docs: steps 08 and 09`.
