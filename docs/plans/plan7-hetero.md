# Plan 7: B2 bipartite patient--code heterogeneous GNN (extra credit, Option A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the design's B2 variant (one patient node carrying DEMO + UTIL linked to the row's code nodes, code--code
edges from Layer U, relation-specific message passing, readout from the patient node) under the identical protocol,
and report it beside the three GNNs and the baselines.

**Architecture:** A fourth `--arch hetero` for step 06 (no new script): `HeteroSubgraphGNN` in `gnn/models.py` with the
same forward signature as `SubgraphGNN`, built from the homogeneous batch: code->code `SAGEConv`, patient->code a
linear message broadcast to the patient's codes, code->patient a mean aggregation into the patient state; relation
messages summed per layer, which is PyG `HeteroConv(..., aggr="sum")` over `SAGEConv`s on the two-type graph
without materialising `HeteroData`. Readout = patient state (no pooling, covariates enter only through the
patient node). Everything else (data, selection, seeds, eval, paired bootstraps) is reused unchanged.

**Tech Stack:** torch 2.15 CPU, torch_geometric 2.8.1, existing `bridgegnn.gnn`.

**Spec:** `docs/CLOSING-DESIGN.md` 5.5; `docs/DESIGN.md` section 6 "Bipartite variant (experiment B2)".

## Global Constraints

- Same rows, split, seeds, grid, selection and prior correction as every other step 06 run.
- No change to the three existing architectures' code paths (G-* numbers must not move).
- Reported honestly whatever the result; compared against the GNNs and both pairing baselines in the same tables.
- Real runs only after the fresh-clone re-run has finished (CPU), in the re-run clone after `git pull`; recorded in
  `REPRODUCTION.md` as the second commit of the submitted artifacts.

## Review Focus

1. `cov_dim == 0` (features without DEMO/UTIL): the patient node needs an initial state; use a learnable vector.
   (Task 1 test `test_hetero_without_covariates`.)
2. Virtual-only rows (no codes): patient state must still be defined and finite. (Task 1 test
   `test_hetero_virtual_only_rows`.)
3. `--weighted 1` with `hetero`: no effect (SAGE); the existing note already covers non-GCN archs. (Task 2 dry-run step.)
4. B2 in the paired bootstraps against both baselines and against G-sage. (Task 2 config step.)

---

### Task 1: `HeteroSubgraphGNN` and the `build_model` factory

**Files:** Modify `src/bridgegnn/gnn/models.py`, `src/bridgegnn/gnn/train.py:77`; Test `tests/test_gnn_models.py`, `tests/test_gnn_train.py`.

**Interfaces:**
- `models.build_model(**model_kwargs) -> nn.Module`: `arch in {gcn, sage, gat}` -> `SubgraphGNN`, `arch == "hetero"` -> `HeteroSubgraphGNN`; same kwargs.
- `HeteroSubgraphGNN.forward(x, edge_index, edge_weight, node_ids, batch, cov) -> logits [n_graphs]`.
  Layer $l$: `h_c' = relu(SAGE_cc(h_c, edge_index) + W_pc p[batch])`, `p' = relu(W_pp p + W_cp mean_pool(h_c, batch))`,
  dropout after each; `p_0 = W_0 cov` (or a learnable vector when `cov_dim == 0`), `h_c,0 = [x, embedding(node_ids+1)]`;
  output `head(p)` with the same two-layer head shape as `SubgraphGNN` minus the covariate concatenation.

- [ ] **Step 1: Failing tests**
  - `test_gnn_models.py`: extend the `arch` parametrisation with `"hetero"` for `test_forward_shapes_and_finite`; add
    `test_hetero_virtual_only_rows` (batch of virtual-only rows -> finite logits of shape [n]) and
    `test_hetero_without_covariates` (`cov_dim=0` -> finite logits); add `test_build_model_dispatch`.
  - `test_gnn_train.py`: `test_fit_hetero_returns_finite_predictions` (setup kwargs with `arch: hetero`, 2 epochs).
- [ ] **Step 2: Run** -> FAIL (`AssertionError: arch in {...}` / no `build_model`).
- [ ] **Step 3: Implement** the class and factory; `train.fit` uses `build_model`.
- [ ] **Step 4: Run** -> PASS; full suite green; ruff clean.
- [ ] **Step 5: Commit** `gnn: bipartite patient-code heterogeneous variant (--arch hetero) for experiment B2`.

### Task 2: Wire B2 through the CLI, grid, pairing and docs

**Files:** `scripts/06_gnn.py:35` (`choices` + hetero), `config/experiments.yaml` (B2 entry replaces the 06b comment),
`config/p2.yaml` (`paired_gnn_runs` += B2; `paired_runs` += `[B2, G-sage]`), `docs/INTERFACES.md`, `docs/DESIGN.md` §6 note,
`scripts/08_report_baselines.py` (`GNNS` += `B2: "Bipartite (hetero)"` so B2 appears in the main rows and CI rows),
`tests/test_driver.py` (`test_grid_contains_b2`).

- [ ] **Step 1: Failing test**: `test_grid_contains_b2` loads `config/experiments.yaml` and asserts one entry `id == "B2"`,
  `base == "gnn"`, `overrides["arch"] == "hetero"`, `family == "B"`.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** the config, CLI choice, report-script label and docs.
- [ ] **Step 4: Dry run**: `uv run python scripts/06_gnn.py --run-id B2 --target cardiometabolic --arch hetero --dry-run` prints `arch: hetero`;
  `07_experiments.py --dry-run | grep -c " B2 "` -> 3. Smoke: `06_gnn.py --run-id SMOKE-B2 --target cardiometabolic --arch hetero --quick --seeds 0` exits 0; remove its outputs and result row.
- [ ] **Step 5: Suite + ruff; commit** `grid: B2 bipartite hetero run, paired against the baselines and G-sage`; push.

### Task 3 (after the re-run finishes): run B2 and fold it into the artifacts and report

- [ ] In `../project2-bridge-gnn-rerun`: `git pull`; `uv run python scripts/07_experiments.py --ids B2` (3 targets x 5 seeds, ~15 min);
  `./scripts/run_all.sh --from 08` (evaluation, fragments, collect, macros).
- [ ] Plan 5 Task 7 comparison and `REPRODUCTION.md` record both commits (grid at e98d56e, B2 at the Task 2 commit).
- [ ] Report: section 4 one paragraph on the heterogeneous variant with citations (Schlichtkrull et al. relational GCN;
  Hu et al. heterogeneous graph transformer; a patient--code bipartite EHR GNN already in the bib if available); table
  rows (main table via the regenerated fragment, ablation table B row); section 6 B paragraph with `\pd{B2}{G-sage}{...}`;
  conclusion sentence replaces "designed but not run". Abstract mentions four GNNs.
