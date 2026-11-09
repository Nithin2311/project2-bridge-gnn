# Plan 3: GNN on patient subgraphs (step 06)

**Goal:** Train and evaluate GCN, GraphSAGE and GAT on per-row patient subgraphs of any layer (U, U-bb, U-bh, D,
N-*), with GroupKFold selection inside the development set, refit, one test pass per seed, and every flag the
experiment grid needs (feature groups, lookback, expansion, weights, depth, aggregator, heads, hidden size,
subsampling, null graphs, bridge-code removal), writing `predictions` and `results` per the contract.

**Architecture:** Three layers of code. `gnn/data.py` turns `rows` + `row_codes` + a node feature matrix + a
layer graph into plain-numpy `Example`s (testable without torch). `gnn/models.py` holds one `SubgraphGNN`
module parameterised by architecture. `gnn/train.py` runs the protocol (fold loop, early stopping, refit,
predict) and writes outputs. `scripts/06_gnn.py` is the CLI from `docs/INTERFACES.md`. Because Dylan's step 02
may not have landed, `nodefeat.py` builds the node feature matrix from Project 1 files the same way Plan D
specifies (Nithin's cross-check copy); step 06 uses `outputs/features/` when it exists and
`outputs/features_xcheck/` otherwise, and logs which.

**Tech Stack:** torch >= 2.4 (CPU), torch_geometric >= 2.6 (GCNConv, SAGEConv, GATConv, global_mean_pool,
global_max_pool, DataLoader), scikit-learn (GroupKFold, metrics), polars, numpy. No MPS (small graphs, CPU is
fine and deterministic).

**Spec:** `docs/DESIGN.md` sections 5, 6, 7, 8; `docs/INTERFACES.md` (step 06 CLI); `docs/DATA-CONTRACT.md`.

## Global Constraints

- Development rows = `split in [train, val]`; test rows touched once per final configuration per seed. Scalers
  for row covariates are fit on the training folds (selection) or on all development rows (refit); never on test.
- Row inputs: codes in the lookback window, `in_main_graph`, not `masked_<target>`; with `--drop-codes bridge`
  also not in the target's B-flag set. A row with no remaining codes gets one virtual node (all-zero features,
  `is_virtual = 1`) so pooling is defined.
- Node feature groups and their meaning in the GNN: CODE = learnable node-id embedding (dim `gnn.embedding_dim`);
  STRUCT, BRIDGE, CHAPTER, COMMUNITY = static columns from the node matrix; TEMP = per-row per-node columns
  `log1p(days_since_last)`, `log1p(days_since_first)`, `recent` (days_since_last <= `windows.recent_days`);
  DEMO = `age_at_index` (standardised), `female`; UTIL = standardised `n_codes_lb*`, `n_rows_lb1095`,
  `n_visit_days_365`, `days_since_first_dx`, `index_year`. `--features` lists any subset of CODE, DEMO, UTIL,
  TEMP, STRUCT, BRIDGE; CHAPTER and COMMUNITY travel with STRUCT.
- Every node always carries two flags: `is_virtual`, `is_context` (1-hop expansion nodes).
- Edges: induced subgraph of the row's node set on the chosen layer. `--weighted 1` passes the layer's edge
  `weight` as `edge_weight` to GCN (SAGE/GAT ignore it, documented); `--weighted 0` passes ones. Layer D is
  directed: SAGE/GAT consume `edge_index` as is; GCN gets the symmetrised edge set and the results row says so.
- Protocol: hyperparameters selected once per (run_id, target) on the selection seed (`gnn.selection_seed = 0`)
  by 5-fold GroupKFold over development rows grouped by `person_id`, early stopping on fold-validation AUPRC
  (`gnn.patience 20`, `gnn.max_epochs 300`), score = mean out-of-fold AUPRC; grid = `gnn.lr_grid x
  gnn.weight_decay_grid`. Then for each seed: refit on all development rows for `round(mean best epoch)` epochs,
  predict test. Predictions file per seed = dev rows with the selection run's out-of-fold `p` + test rows from
  that seed's refit. Results row per seed per the contract; `params_json` holds the best grid point, the epoch
  count and all CLI flags.
- Loss: `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` on the training rows of the fold or refit. Adam.
  Batch size `gnn.batch_size 64`. Seeds set torch, numpy and python RNGs; `torch.use_deterministic_algorithms`
  not required on CPU.
- Config `gnn:` section with reasons: `hidden 64, depth 2, dropout 0.3, heads 4, agg mean, batch_size 64,
  lr_grid [1e-3, 3e-4], weight_decay_grid [1e-4], max_epochs 300, patience 20, embedding_dim 16,
  selection_seed 0, device cpu, quick: {max_epochs 5, patience 2}` (`--quick` for smoke runs).
- Outputs: `outputs/predictions/{run_id}_{target}_seed{seed}.parquet`, a row appended to
  `outputs/results/gnn.csv` (read-concat-write through `contract`), curves JSON
  `outputs/logs/06_{run_id}_{target}_seed{seed}.json` (per-epoch train loss and val AUPRC for the refit and
  for each fold of the selection), log `outputs/logs/06_{run_id}_{target}.log`, one JSON summary line per seed
  on stdout. `--dry-run` prints the resolved configuration (incl. which node-feature dir) and exits 0.

## Review Focus

1. A development row whose patient also has rows in another fold must never leak: GroupKFold by `person_id`
   is asserted in a test that checks no `person_id` spans a training and a held-out fold. Test in Task 4.
2. With `--features CODE` only, the node input is the embedding plus the two flags and nothing else; with no
   CODE, there is no embedding layer. Test in Task 2 (column accounting) and Task 3 (module shapes).
3. `--drop-codes bridge` removes the target's B-flag codes from the subgraph AND from `n_codes` style UTIL
   covariates for that row. Test in Task 2.
4. A row with zero codes after masking yields exactly one virtual node and no edges, and training does not
   crash on a batch of such rows. Test in Task 2 and Task 4.
5. Results and predictions written by two different seeds must not overwrite each other, and re-running the
   same (run_id, target, seed) replaces the earlier results row instead of duplicating it. Test in Task 4.

---

### Task 1: Dependencies and node feature matrix (`nodefeat.py`)

**Files:**
- Modify: `pyproject.toml` (torch, torch_geometric already in the `gnn` extra; make `uv sync --extra gnn` the
  documented step), `config/p2.yaml` (`gnn:` section)
- Create: `src/bridgegnn/nodefeat.py`
- Test: `tests/test_nodefeat.py`

**Interfaces:**
- Consumes: P1 `p2/node_features.csv`, `bridges.csv`, `graph_nodes.csv`; `contract.TARGETS`.
- Produces:
  - `nodefeat.STRUCT_COLS = ["prevalence", "degree", "weighted_degree", "betweenness", "bt_null_percentile",
    "closeness", "eigenvector", "pagerank", "participation", "within_module_z", "degree_residual_z",
    "inpatient_share", "ccir_chronic_share", "mean_age_first_occurrence", "female_share", "backbone_degree"]`.
  - `nodefeat.build(node_features: pl.DataFrame, bridges: pl.DataFrame, targets: list[str], min_community_size: int = 5)
    -> tuple[pl.DataFrame, np.ndarray, dict[str, list[str]]]`: `(node_index, X, groups)` where `node_index` matches
    the contract (categories sorted, `idx` 0..n-1), `X` is float32 `[n, F]` with columns standardised (mean 0,
    sd 1 over the n nodes; nulls filled with the column mean before scaling), and `groups` maps `STRUCT`,
    `BRIDGE`, `CHAPTER`, `COMMUNITY` to column-name lists in `X` order. BRIDGE columns: `bridge_flag`,
    `bridge_flag_<t>`, `bridge_score_<t>` for each target (per the design, section 5: survivor flag per target =
    survivor on any pair containing the target; score = max `bridge_betweenness` over pairs containing it).
    CHAPTER = one-hot of the chapter letter; COMMUNITY = one-hot of `community_seed0` for communities with at
    least `min_community_size` nodes plus `community_other`.
  - `nodefeat.bflag_codes(node_features, bridges, target) -> set[str]`: the target's B-flag set (used by
    `--drop-codes bridge`).
  - `nodefeat.write(node_index, X, groups, out_dir) -> None` and `nodefeat.read(dir) -> (node_index, X, groups)`
    using `node_index.csv`, `node_x.npy`, `feature_groups.json` (the step 02 artifact names).

- [ ] **Step 1: Write failing tests** `tests/test_nodefeat.py` (a `node_feats` fixture synthesised from
  `synth.nodes` with the STRUCT columns random, `community_seed0` in {0, 1, 2, 7}, `bridge_survivor` True for
  two codes, `chapter` from the synth node table):

```python
def test_build_shapes_and_groups(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    assert X.shape[0] == idx.height == synth.nodes.height and X.dtype == np.float32
    assert set(groups) == {"STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY"}
    assert sum(len(v) for v in groups.values()) == X.shape[1]
    assert abs(float(X[:, 0].mean())) < 1e-5 and abs(float(X[:, 0].std()) - 1) < 1e-3

def test_bridge_flag_per_target_is_union_over_pairs(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    col = groups["BRIDGE"].index("bridge_flag_mental_health")
    flagged = set(idx.filter(pl.Series(X[:, col] > 0))["category"])
    expected = set(synth.bridges.filter(pl.col("bridge_pass") & pl.col("pair").str.contains("mental_health"))["category"]) & set(node_feats["category"])
    assert flagged == expected

def test_bflag_codes_matches_matrix(node_feats, synth):
    assert nodefeat.bflag_codes(node_feats, synth.bridges, "cardiometabolic") <= set(node_feats["category"])

def test_write_read_roundtrip(tmp_path, node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    nodefeat.write(idx, X, groups, tmp_path)
    idx2, X2, groups2 = nodefeat.read(tmp_path)
    assert idx2.equals(idx) and np.allclose(X, X2) and groups2 == groups
```

- [ ] **Step 2: Run** `uv run pytest tests/test_nodefeat.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `nodefeat.py`; add the `gnn:` config section.
- [ ] **Step 4: Run** tests and the whole suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: node feature matrix builder (step 02 cross-check) and gnn config`.

---

### Task 2: Examples from rows (`gnn/data.py`, numpy only)

**Files:**
- Create: `src/bridgegnn/gnn/__init__.py`, `src/bridgegnn/gnn/data.py`
- Test: `tests/test_gnn_data.py`

**Interfaces:**
- Produces:
  - `@dataclass Example: x: np.ndarray [n_i, F_total] float32; edge_index: np.ndarray [2, m_i] int64;
    edge_weight: np.ndarray [m_i] float32; node_ids: np.ndarray [n_i] int64 (node_index idx, -1 for virtual);
    cov: np.ndarray [C] float32; y: int; person_id: int; index_date: date; split: str`.
  - `@dataclass Options: layer: str; features: list[str]; lookback: int; expand: bool; weighted: bool;
    drop_codes: set[str]; directed: bool; symmetrise: bool` (built by the CLI; `symmetrise = directed and arch == "gcn"`).
  - `data.feature_columns(groups: dict, features: list[str], recent_days: int) -> dict`: returns
    `{"static_cols": [indices into X], "temp": bool, "flags": ["is_virtual", "is_context"], "n_static": int,
    "n_temp": int}`; TEMP contributes 3 columns when present.
  - `data.build_examples(rows_t: pl.DataFrame, row_codes: pl.DataFrame, target: str, node_index: pl.DataFrame,
    X: np.ndarray, groups: dict, g_layer: nx.Graph | nx.DiGraph, opts: Options, cfg: dict) -> list[Example]`:
    one Example per row of `rows_t` (already filtered to the target); codes selected by lookback flag
    (`in_lb365`, `in_lb1095`, or all), `in_main_graph`, not `masked_<target>`, not in `opts.drop_codes`;
    node order = sorted `idx`; `x = [static cols | temp cols | is_virtual | is_context]`; expansion adds every
    layer-graph neighbour of the row's codes not already present, with TEMP zeros and `is_context = 1`;
    edges = induced subgraph on the layer (both directions for undirected layers; as stored for directed;
    plus reversed copies when `opts.symmetrise`); `edge_weight` from the layer's `weight` attr or ones.
    `cov` = the DEMO and UTIL columns requested (raw; scaling happens in train), with UTIL counts recomputed
    from the kept codes when `drop_codes` is non-empty (`n_codes_*` minus dropped codes present).
  - `data.covariate_names(features) -> list[str]`.
  - `data.fit_cov_scaler(examples, names) -> (mean, sd)` over the given examples and `apply_cov_scaler`.

- [ ] **Step 1: Write failing tests** `tests/test_gnn_data.py` (uses Plan 1's `rows`/`row_codes` on synth plus
  `nodefeat.build`):

```python
def test_example_count_and_shapes(built):
    exs = data.build_examples(rows_t, rc, "mental_health", idx, X, groups, synth.graph, opts_all, cfg)
    assert len(exs) == rows_t.height
    e = exs[0]; assert e.x.shape[1] == n_static + 3 + 2 and e.edge_index.shape[0] == 2 and e.cov.shape[0] == len(data.covariate_names(opts_all.features))

def test_masked_and_dropped_codes_absent(built):
    ... build with drop_codes = {one present code}; assert no example contains that node id and n_codes cov decreased for rows that had it

def test_code_only_features_leave_only_flags(built):
    opts = replace(opts_all, features=["CODE"]); e = data.build_examples(...)[0]
    assert e.x.shape[1] == 2   # is_virtual, is_context only; the embedding is added by the model from node_ids

def test_empty_row_gets_virtual_node(built):
    ... pick a row, drop all its codes via drop_codes; e.x.shape[0] == 1 and e.node_ids[0] == -1 and e.edge_index.shape[1] == 0 and e.x[0, -2] == 1

def test_expand_adds_context_nodes_with_flag(built):
    e0 = ...expand False; e1 = ...expand True (same row); assert e1.x.shape[0] >= e0.x.shape[0] and (e1.x[:, -1] == 1).sum() == e1.x.shape[0] - e0.x.shape[0]

def test_directed_layer_symmetrised_only_for_gcn(built):
    dg = a DiGraph with one edge (a -> b) among the row's codes
    e_dir = ...opts(directed=True, symmetrise=False); e_sym = ...symmetrise=True
    assert e_dir.edge_index.shape[1] == 1 and e_sym.edge_index.shape[1] == 2

def test_edge_weight_ones_when_unweighted(built):
    assert np.all(e_unweighted.edge_weight == 1.0) and not np.all(e_weighted.edge_weight == 1.0)

def test_cov_scaler_fit_only_on_given_examples(built):
    mean, sd = data.fit_cov_scaler(dev_examples, names); assert mean.shape == (len(names),)
```

- [ ] **Step 2: Run**. Expected: FAIL. **Step 3: Implement.** **Step 4: Run** tests + suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: per-row subgraph examples with feature groups, expansion, masking`.

---

### Task 3: Model (`gnn/models.py`)

**Files:**
- Create: `src/bridgegnn/gnn/models.py`
- Test: `tests/test_gnn_models.py` (skipped when torch is not importable)

**Interfaces:**
- Produces: `models.SubgraphGNN(arch: str, in_dim: int, cov_dim: int, hidden: int, depth: int, dropout: float,
  heads: int, agg: str, n_nodes: int, embedding_dim: int, use_embedding: bool)` with
  `forward(x, edge_index, edge_weight, node_ids, batch, cov) -> logits [B]`. Node input = `x` concatenated with
  `Embedding(n_nodes + 1, embedding_dim)` looked up on `node_ids + 1` (so virtual = index 0) when
  `use_embedding`. Layers: GCNConv (uses `edge_weight`), SAGEConv(aggr=agg), GATConv(heads, hidden // heads,
  concat) x `depth`, ReLU + dropout between; readout = `[global_mean_pool | global_max_pool]`; head =
  `Linear(2*hidden + cov_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)`.
  `models.to_pyg(examples: list[Example]) -> list[torch_geometric.data.Data]` with attributes `x, edge_index,
  edge_weight, node_ids, cov (shape [1, C]), y (float), row_pos (int)`.

- [ ] **Step 1: Write failing tests**:

```python
@pytest.mark.parametrize("arch", ["gcn", "sage", "gat"])
def test_forward_shapes_and_finite(arch, examples_small):
    m = models.SubgraphGNN(arch, in_dim, cov_dim, 16, 2, 0.1, 4, "mean", n_nodes, 8, True)
    batch = next(iter(DataLoader(models.to_pyg(examples_small), batch_size=4)))
    out = m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    assert out.shape == (4,) and torch.isfinite(out).all()

def test_no_embedding_when_code_off(...):
    m = models.SubgraphGNN("sage", in_dim, cov_dim, 16, 2, 0.1, 4, "mean", n_nodes, 8, use_embedding=False)
    assert m.embedding is None

def test_sage_max_and_gat_heads_configurable(...):  # agg="max", heads=2 both construct and run
def test_batch_of_virtual_only_rows_runs(...):      # examples with 1 virtual node each, no edges
```

- [ ] **Step 2: Run** (FAIL). **Step 3: Implement.** **Step 4: Run** (PASS). **Step 5: Commit** `feat: SubgraphGNN (GCN/SAGE/GAT) with pooling and covariate head`.

---

### Task 4: Training protocol and outputs (`gnn/train.py`)

**Files:**
- Create: `src/bridgegnn/gnn/train.py`
- Test: `tests/test_gnn_train.py` (torch required; uses `--quick`-sized epochs)

**Interfaces:**
- Produces:
  - `train.set_seed(seed)`.
  - `train.fit(train_ex, val_ex, params: dict, seed: int, cfg_gnn: dict, model_kwargs: dict) ->
    tuple[SubgraphGNN, int, dict]` returning the model at its best epoch (state restored), the best epoch, and
    curves `{"train_loss": [...], "val_auprc": [...]}`; early stopping on val AUPRC with `patience`; when
    `val_ex` is empty (refit) trains exactly `params["epochs"]` epochs.
  - `train.predict(model, examples) -> np.ndarray` of probabilities.
  - `train.group_folds(examples, n_folds, seed) -> list[tuple[np.ndarray, np.ndarray]]` using
    `sklearn.model_selection.GroupKFold` on `person_id` (shuffled group order by seed).
  - `train.select(dev_ex, grid: list[dict], cfg_gnn, model_kwargs, seed) -> tuple[dict, float, np.ndarray, int, list]`:
    best params, best mean OOF AUPRC, OOF probabilities aligned to `dev_ex`, mean best epoch, per-fold curves.
    Folds with a single class in validation are skipped with a warning.
  - `train.run(run_id, target, dev_ex, test_ex, grid, seeds, cfg, model_kwargs, flags: dict, out_dir) -> list[dict]`:
    selection on `cfg.gnn.selection_seed`, then per seed refit + test predict; writes predictions parquet (dev
    OOF + test), curves JSON, appends/replaces the results row (keyed by run_id, target, seed) in
    `outputs/results/gnn.csv`; returns the results dicts. Covariate scaler fit on training folds / dev only.
    Metrics: `roc_auc_score`, `average_precision_score`, `brier_score_loss` on test.

- [ ] **Step 1: Write failing tests** (all with `cfg_gnn = {"max_epochs": 3, "patience": 2, "batch_size": 8, ...}`):

```python
def test_group_folds_never_split_a_patient(dev_examples):
    for tr, va in train.group_folds(dev_examples, 3, 0):
        assert not ({dev_examples[i].person_id for i in tr} & {dev_examples[i].person_id for i in va})

def test_fit_returns_finite_predictions_and_curves(dev_examples):
    m, best, curves = train.fit(dev_examples[:30], dev_examples[30:40], {"lr": 1e-3, "weight_decay": 0.0}, 0, quick_cfg, kwargs)
    p = train.predict(m, dev_examples[30:40]); assert p.shape == (10,) and np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1
    assert len(curves["train_loss"]) == len(curves["val_auprc"]) >= 1 and 1 <= best <= len(curves["train_loss"])

def test_refit_trains_fixed_epochs(dev_examples):
    m, best, curves = train.fit(dev_examples, [], {"lr": 1e-3, "weight_decay": 0.0, "epochs": 2}, 0, quick_cfg, kwargs)
    assert best == 2 and len(curves["train_loss"]) == 2

def test_run_writes_contract_files_and_replaces_rows(tmp_path, dev_examples, test_examples):
    res = train.run("T-sage", "mental_health", dev_examples, test_examples, [{"lr": 1e-3, "weight_decay": 0.0}], [0, 1], cfg_quick, kwargs, flags, tmp_path)
    assert len(res) == 2
    preds = contract.read(tmp_path / "predictions" / "T-sage_mental_health_seed0.parquet", "predictions")
    assert set(preds["split"]) >= {"test"} and preds.filter(pl.col("split") == "test").height == len(test_examples)
    results = contract.read(tmp_path / "results" / "gnn.csv", "results"); assert results.height == 2
    train.run(... seeds [0] ...); assert contract.read(tmp_path / "results" / "gnn.csv", "results").height == 2   # replaced, not duplicated

def test_run_handles_virtual_only_batch(...):  # dev examples all emptied -> runs without error
def test_select_skips_single_class_fold(...):  # construct folds where one validation fold has no positives -> warning, finite score
```

- [ ] **Step 2: Run** (FAIL). **Step 3: Implement.** **Step 4: Run** tests + suite (PASS).
- [ ] **Step 5: Commit** `feat: GNN training protocol with GroupKFold selection, refit, contract outputs`.

---

### Task 5: CLI `scripts/06_gnn.py`, smoke run, real run of the main comparison

**Files:**
- Create: `scripts/06_gnn.py`
- Modify: `docs/INTERFACES.md` (any flag the implementation added: `--quick`, `--node-features-dir`), `README.md`.

**Interfaces:**
- CLI exactly as `docs/INTERFACES.md` plus `--quick` and `--node-features-dir`. Resolution order for node
  features: `--node-features-dir` if given; else `outputs/features/` if `node_x.npy` exists (Dylan's step 02);
  else build with `nodefeat.build` into `outputs/features_xcheck/` and use it. The chosen directory is printed
  and stored in `params_json`.
- Layer loading: `U` = P1 `graph_main.graphml`; `U-bb` = P1 `p2/graph_backbone_topk.graphml`; `U-bh` =
  `outputs/layers/graph_bh.graphml`; `D` = `outputs/layers/graph_directed.graphml` (directed); `N` =
  `outputs/layers/null/{null-graph}.graphml`.
- `--subsample-file` template with `{seed}`: development rows restricted to the listed patients for that seed
  (selection uses the selection seed's file).
- `family` for the results row: looked up in `config/experiments.yaml` by run_id; `gnn` for `G-*`; else `gnn`.

- [ ] **Step 1:** Implement the CLI (argparse; `--dry-run` prints the resolved config as JSON and exits 0).
- [ ] **Step 2: Smoke** `uv run python scripts/06_gnn.py --run-id SMOKE --target mental_health --arch sage --quick --seeds 0`.
  Expected: finishes in under 3 minutes, writes predictions and a results row, prints one JSON line. Then
  `--dry-run` for `--layer D --arch gcn` prints `symmetrise: true`. Delete the SMOKE outputs.
- [ ] **Step 3: Real run, main comparison**, in the background, logged:
  `for a in gcn sage gat; do for t in cardiometabolic cancer_hematologic mental_health; do uv run python scripts/06_gnn.py --run-id G-$a --target $t --arch $a; done; done`.
  Expected: 9 (arch, target) runs x 5 seeds = 45 results rows in `outputs/results/gnn.csv`; wall time a few
  hours on the M4. Record per-run time in the ledger.
- [ ] **Step 4: Commit** `feat: step 06 CLI; main GNN comparison run`.

---

### Task 6: Docs

- [ ] Update `README.md` (step 06 status, `uv sync --extra gnn`), `docs/DATA-CONTRACT.md` (curves JSON,
  `features_xcheck/`), `docs/INTERFACES.md` (added flags, family lookup). Commit `docs: step 06`.
