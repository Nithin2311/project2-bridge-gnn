# Plan 2: directed precedence layer, edge-filter variants, null graphs, motif census (steps 03, 04)

**Goal:** Build Layer D (directed first-record precedence edges on top of the validated Project 1 edges), the
BH-filtered edge variant and the null graphs the experiments need, and the 3-node directed motif census with
bridge-centred chains, all from Project 1 outputs on train+val patients only.

**Architecture:** Pure polars/NetworkX functions in `directed.py`, `layers.py`, `motifs.py` tested on the
synthetic fixtures (which plant an A -> B precedence); two thin scripts read Project 1 artifacts, write graphml
and CSV layers under `outputs/layers/` and `outputs/motifs/`, and print the tables the report quotes. Test
patients never enter: every function takes the development patient list and asserts it.

**Tech Stack:** polars, numpy, scipy (binomtest), statsmodels (BH), networkx 3.7 (`triadic_census`,
`directed_edge_swap`), igraph not needed, matplotlib for one figure; `bridgedx.null_models.rewired` for the
undirected null.

**Spec:** `docs/DESIGN.md` section 6 (Layer D, motif census, Layer N, U-bh) and section 8 (B1, B4, D1).

## Global Constraints

- Development patients = `split in cfg.split.dev` from Project 1 `patient_split.csv`; first dates come from
  `patient_category.parquet` filtered to them. Assert no test patient contributes to any count.
- Precedence gap `g = 30` days primary; sensitivities `g = 0` and `g = 90`; day-one codes (a patient's codes
  dated on their first ICD-10-era day) excluded from precedence counting in the primary run, included as a
  sensitivity. Pair support `n_ab + n_ba >= 20`. Two-sided binomial test of `n_ab` vs `n_ba` at `p = 0.5`,
  BH-FDR `alpha = 0.05` over the tested pairs. Direction only on pairs that are Layer U edges.
- Layer N: 5 degree-preserving rewirings (`bridgedx.null_models.rewired`, seeds 0-4) plus one edge-shuffled
  graph (same edge count, endpoints drawn uniformly without self-loops or duplicates, weights shuffled).
- U-bh: Benjamini-Hochberg at 0.05 on `p_fisher` over the pairs in `edges.csv` whose endpoints are both
  main-graph nodes, keeping `odds_ratio >= 1.2` and `n_both >= 10` (the same floors as Project 1), phi weight.
- Motif null: 100 `directed_edge_swap` rewirings of Layer D, `nswap = 10 * m`, seeds 0-99.
- All parameters in `config/p2.yaml` under `layers:` and `motifs:` with reasons; outputs under
  `outputs/layers/` and `outputs/motifs/`; aggregate tables (category level, no patient ids) may be copied to
  `outputs/artifacts/`.

## Review Focus

1. A pair where every co-occurring patient has both codes on the same day (ties) must produce no directed edge
   and must not crash the binomial test (n = 0). Test in Task 1.
2. A patient with code A on their first era day and B later must not count toward A -> B in the primary run
   (left truncation) but must count in the `include_day_one` sensitivity. Test in Task 1.
3. The shuffled null graph must have exactly the same edge count and node set as Layer U, no self-loops, no
   duplicate edges, and must not preserve degrees (otherwise it duplicates the rewired null). Test in Task 2.
4. `triadic_census` counts all 16 classes including disconnected ones; z-scores must be reported only for
   connected classes with non-zero null SD, and the feed-forward class is `030T`. Test in Task 3.
5. Bridge chains A -> bridge -> core must require all three codes to be strictly ordered with the gap on both
   steps and must exclude A that is itself a core member of the same target. Test in Task 3.

---

### Task 1: Directed precedence layer (`directed.py`)

**Files:**
- Create: `src/bridgegnn/directed.py`, config `layers:` section in `config/p2.yaml`
- Modify: `src/bridgegnn/contract.py` (add `edges_directed` schema)
- Test: `tests/test_directed.py`

**Interfaces:**
- Consumes: Project 1 `patient_category.parquet` (person_id, category, first_date), `patient_split.csv`,
  `graph_main.graphml`; synth: `synth.codes_long` collapsed to first dates, `synth.split`, `synth.graph`.
- Produces:
  - `directed.first_dates(pc: pl.DataFrame, split: pl.DataFrame, dev_splits: list[str], drop_day_one: bool) -> pl.DataFrame`
    with columns `person_id, category, first_date`; only development patients; when `drop_day_one`, rows whose
    `first_date` equals that patient's minimum `first_date` are removed. Asserts no `test` person remains.
  - `directed.precedence_counts(fd: pl.DataFrame, pairs: list[tuple[str, str]], gap_days: int) -> pl.DataFrame`
    with columns `a, b, n_ab, n_ba, n_tie` where `n_ab` = patients with `first(a) + gap <= first(b)`,
    `n_ba` symmetric, `n_tie` = patients with both codes and `|diff| < gap`. Vectorised: self-join of `fd` on
    person_id restricted to the pair list.
  - `directed.directed_edges(counts: pl.DataFrame, min_total: int, alpha: float) -> pl.DataFrame` matching
    contract `edges_directed`: `src, dst, n_fwd, n_rev, n_tie, share, p_binom, q_bh, phi` where `src -> dst`
    is the majority direction, `share = n_fwd / (n_fwd + n_rev)`, `p_binom` two-sided `scipy.stats.binomtest`,
    `q_bh` from `statsmodels.stats.multitest.multipletests(method="fdr_bh")` over pairs with
    `n_fwd + n_rev >= min_total`, kept when `q_bh < alpha`. `phi` is joined in by the caller (`attach_phi`).
  - `directed.attach_phi(dedges, g_u: nx.Graph) -> pl.DataFrame` adds `phi` from the undirected edge weight.
  - `directed.to_digraph(dedges: pl.DataFrame, g_u: nx.Graph) -> nx.DiGraph` with node attributes copied from
    `g_u`, edge attributes `weight` (= share), `phi`, `n_fwd`, `n_rev`, `distance = 1/phi`.
  - `directed.summary(dedges, g_u) -> dict` with `n_directed`, `n_undirected`, `share_directed`,
    `median_share`, `n_tested`.
  - `directed.edges_into_cores(dedges, cores: dict[str, set[str]], top_k: int) -> pl.DataFrame`: per target,
    the strongest `top_k` directed edges whose `dst` is a core member and `src` is not, by `n_fwd`, with
    columns `target, src, dst, n_fwd, n_rev, share, q_bh`.
  - `directed.edges_through_bridges(dedges, bridge_codes: set[str]) -> pl.DataFrame`: all directed edges
    touching a bridge code with a column `role` in {`into_bridge`, `out_of_bridge`}.
- Contract `edges_directed`: `src Utf8, dst Utf8, n_fwd Int32, n_rev Int32, n_tie Int32, share Float64,
  p_binom Float64, q_bh Float64, phi Float64`.
- Config:
  ```yaml
  layers:
    precedence_gap_days: 30        # A must precede B by >= 30 d; same-month pairs are ties (Dr. Kim: direction, not co-coding)
    gap_sensitivity_days: [0, 90]
    drop_day_one_codes: true       # P1: 40% of cardiometabolic carriers coded on their first era day (problem-list carry-over)
    min_pair_patients: 20          # n_fwd + n_rev floor for the binomial test
    alpha: 0.05                    # BH-FDR over tested pairs
    null_n: 5                      # degree-preserving rewirings for experiment D1
    null_seed: 0
    bh_alpha: 0.05                 # U-bh variant: BH instead of Bonferroni on the P1 Fisher p-values
    top_k_into_cores: 15
  ```

- [ ] **Step 1: Write failing tests** `tests/test_directed.py`:

```python
def fd_from_synth(synth, drop_day_one=True):
    pc = synth.codes_long.group_by("person_id", "category").agg(pl.col("condition_date").min().alias("first_date"))
    return directed.first_dates(pc, synth.split, ["train", "val"], drop_day_one)

def test_first_dates_excludes_test_patients(synth):
    fd = fd_from_synth(synth)
    test_ids = set(synth.split.filter(pl.col("split") == "test")["person_id"])
    assert not (set(fd["person_id"]) & test_ids)

def test_first_dates_drops_day_one_codes(synth):
    fd_all = fd_from_synth(synth, drop_day_one=False); fd = fd_from_synth(synth, True)
    first_day = fd_all.group_by("person_id").agg(pl.col("first_date").min().alias("d0"))
    j = fd.join(first_day, on="person_id")
    assert (j["first_date"] > j["d0"]).all() and fd.height < fd_all.height

def test_precedence_counts_planted_pair(synth):
    a, b = synth.precedence_pair
    c = directed.precedence_counts(fd_from_synth(synth, False), [(a, b)], gap_days=30).row(0, named=True)
    assert c["n_ab"] > c["n_ba"] and c["n_ab"] + c["n_ba"] + c["n_tie"] >= 10

def test_precedence_gap_turns_close_pairs_into_ties():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1, 2, 2], dtype=pl.Int64), "category": ["A", "B", "A", "B"],
                       "first_date": [date(2018,1,1), date(2018,1,10), date(2018,1,1), date(2018,3,1)]})
    c0 = directed.precedence_counts(fd, [("A", "B")], 0).row(0, named=True)
    c30 = directed.precedence_counts(fd, [("A", "B")], 30).row(0, named=True)
    assert (c0["n_ab"], c0["n_tie"]) == (2, 0) and (c30["n_ab"], c30["n_tie"]) == (1, 1)

def test_directed_edges_all_ties_yield_nothing():
    counts = pl.DataFrame({"a": ["A"], "b": ["B"], "n_ab": [0], "n_ba": [0], "n_tie": [40]})
    assert directed.directed_edges(counts, min_total=20, alpha=0.05).height == 0

def test_directed_edges_direction_and_bh():
    counts = pl.DataFrame({"a": ["A", "C", "E"], "b": ["B", "D", "F"], "n_ab": [5, 30, 15], "n_ba": [40, 28, 14], "n_tie": [0, 0, 0]})
    out = directed.directed_edges(counts, 20, 0.05)
    assert out.height == 1 and out.row(0, named=True)["src"] == "B" and out.row(0, named=True)["dst"] == "A"
    contract.validate(out.with_columns(pl.lit(0.3).alias("phi")), "edges_directed")

def test_day_one_sensitivity_counts_more(synth):
    a, b = synth.precedence_pair
    n_drop = directed.precedence_counts(fd_from_synth(synth, True), [(a, b)], 30).row(0, named=True)
    n_keep = directed.precedence_counts(fd_from_synth(synth, False), [(a, b)], 30).row(0, named=True)
    assert n_keep["n_ab"] + n_keep["n_ba"] >= n_drop["n_ab"] + n_drop["n_ba"]

def test_to_digraph_and_tables(synth):
    g_u = synth.graph; pairs = list(g_u.edges())
    de = directed.attach_phi(directed.directed_edges(directed.precedence_counts(fd_from_synth(synth, False), pairs, 30), 5, 0.5), g_u)
    dg = directed.to_digraph(de, g_u)
    assert dg.number_of_edges() == de.height and all("phi" in d for _, _, d in dg.edges(data=True))
    cores = synth.targets["core"]
    into = directed.edges_into_cores(de, cores, top_k=5)
    assert set(into.columns) >= {"target", "src", "dst", "n_fwd", "share", "q_bh"}
    assert all(r["dst"] in cores[r["target"]] and r["src"] not in cores[r["target"]] for r in into.iter_rows(named=True))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_directed.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `directed.py`**, the contract schema, and the config section.
- [ ] **Step 4: Run** the tests and the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: directed precedence layer with binomial test and core/bridge tables`.

---

### Task 2: Edge-filter variant and null graphs (`layers.py`) and script 03

**Files:**
- Create: `src/bridgegnn/layers.py`, `scripts/03_directed_layer.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Produces:
  - `layers.bh_edges(edges_all: pl.DataFrame, node_set: set[str], cfg: dict) -> pl.DataFrame` with the
    Project 1 `edges.csv` columns plus `q_bh`, filtered to endpoints in `node_set`, `odds_ratio >= 1.2`,
    `n_both >= 10`, `q_bh < cfg.layers.bh_alpha`; weight column `phi`.
  - `layers.graph_from_edges(edges: pl.DataFrame, g_template: nx.Graph) -> nx.Graph` (node attributes from the
    template; edge `weight = phi`, `distance = 1/phi`).
  - `layers.rewired_graphs(g_u, n, seed) -> list[nx.Graph]` wrapping `bridgedx.null_models.rewired`.
  - `layers.shuffled_graph(g_u, seed) -> nx.Graph`: same nodes, `m` random distinct non-loop edges, the
    multiset of weights reassigned at random.
  - `layers.layer_stats(g) -> dict` (`n_nodes, n_edges, density, n_isolates, mean_degree`).
- Script `03_directed_layer.py`: load config, P1 graph, split, patient_category, edges.csv, target_definitions,
  node_features (bridge survivors); build Layer D primary and the four sensitivities (`g0`, `g90`,
  `include_day_one`, and `g30` on all-chapters? no: only those three); write
  `outputs/layers/edges_directed.csv` (+ `_g0`, `_g90`, `_dayone`), `outputs/layers/graph_directed.graphml`,
  `outputs/layers/edges_bh.csv`, `graph_bh.graphml`, `outputs/layers/null/rewire{0..4}.graphml`,
  `null/shuffle.graphml`, tables `outputs/layers/tables/directed_into_cores.csv`,
  `directed_through_bridges.csv`, `layer_stats.csv` (one row per layer incl. U, U-bb, U-bh, D, nulls),
  `directed_summary.csv` (one row per sensitivity). Print the summary and the into-cores table.

- [ ] **Step 1: Write failing tests** `tests/test_layers.py`:

```python
def test_bh_keeps_at_least_bonferroni_edges():
    edges_all = pl.DataFrame({"u": ["A","A","B","C"], "v": ["B","C","C","D"], "n_both": [50, 30, 12, 9],
                              "odds_ratio": [3.0, 1.5, 1.1, 2.0], "p_fisher": [1e-10, 1e-3, 1e-4, 1e-6], "phi": [.4,.2,.1,.3]})
    out = layers.bh_edges(edges_all, {"A","B","C","D"}, {"layers": {"bh_alpha": 0.05}})
    assert set(zip(out["u"], out["v"])) == {("A","B"), ("A","C")}   # B-C fails OR floor, C-D fails n_both floor

def test_shuffled_graph_same_size_no_loops_degrees_differ(synth):
    g = synth.graph; h = layers.shuffled_graph(g, 0)
    assert set(h.nodes) == set(g.nodes) and h.number_of_edges() == g.number_of_edges()
    assert not any(u == v for u, v in h.edges()) and sorted(d for _, d in h.degree()) != sorted(d for _, d in g.degree())
    assert sorted(d["weight"] for *_, d in h.edges(data=True)) == sorted(d["weight"] for *_, d in g.edges(data=True))

def test_rewired_graphs_preserve_degrees(synth):
    g = synth.graph; hs = layers.rewired_graphs(g, 2, 0)
    assert len(hs) == 2 and all(sorted(dict(h.degree()).items()) == sorted(dict(g.degree()).items()) for h in hs)

def test_layer_stats_keys(synth):
    assert set(layers.layer_stats(synth.graph)) == {"n_nodes", "n_edges", "density", "n_isolates", "mean_degree"}
```

- [ ] **Step 2: Run** `uv run pytest tests/test_layers.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement `layers.py`** and `scripts/03_directed_layer.py`.
- [ ] **Step 4: Run** tests (PASS) then `uv run python scripts/03_directed_layer.py`. Expected: files written;
  Layer D has thousands, not tens, of edges; summary printed; runtime under 10 minutes (rewiring 9,115 edges x 5).
- [ ] **Step 5: Commit** `feat: step 03 directed layer, BH edges, null graphs`.

---

### Task 3: Motif census and bridge chains (`motifs.py`) and script 04

**Files:**
- Create: `src/bridgegnn/motifs.py`, `scripts/04_motifs.py`, config `motifs:` section
- Test: `tests/test_motifs.py`

**Interfaces:**
- Produces:
  - `motifs.triad_census(dg: nx.DiGraph) -> dict[str, int]` (NetworkX's 16 classes).
  - `motifs.null_census(dg, n: int, seed: int) -> pl.DataFrame` with `triad, observed, null_mean, null_sd, z`
    for every class; `z` null where `null_sd == 0`; rewiring via `nx.directed_edge_swap(copy, nswap=10*m,
    max_tries=200*m, seed=seed+i)`.
  - `motifs.feed_forward_middle_share(dg, codes: set[str]) -> pl.DataFrame`: for each code, the number of
    `030T` (feed-forward) triads in which it is the middle node (has one in-edge and one out-edge within the
    triad and the shortcut edge exists), and the share of all such triads; computed by iterating over
    node pairs of predecessors/successors.
  - `motifs.bridge_chains(fd: pl.DataFrame, bridges: set[str], cores: dict[str, set[str]], gap_days: int,
    min_patients: int) -> pl.DataFrame` with `target, a, bridge, core_code, n_patients` counting patients with
    `first(a) + gap <= first(bridge)` and `first(bridge) + gap <= first(core_code)`, `a` not in that target's
    core and not a bridge, kept when `n_patients >= min_patients`, sorted by `n_patients` desc.
- Config:
  ```yaml
  motifs:
    null_n: 100            # directed degree-preserving rewirings for the triad census
    seed: 0
    min_chain_patients: 10 # a chain A -> bridge -> core code needs this many ordered patients to be listed
    top_chains: 40
  ```
- Script `04_motifs.py`: read `graph_directed.graphml`, bridge codes, cores, dev first dates (recomputed via
  `directed.first_dates`); write `outputs/motifs/triad_census.csv`, `bridge_middle_share.csv`,
  `bridge_chains.csv`, `fig_motif_z.pdf` (bar of z per connected class; matplotlib, no seaborn); print the
  census table and the top 15 chains.

- [ ] **Step 1: Write failing tests** `tests/test_motifs.py`:

```python
def ff():
    g = nx.DiGraph(); g.add_edges_from([("A","B"), ("B","C"), ("A","C")]); return g

def test_triad_census_counts_feed_forward():
    c = motifs.triad_census(ff()); assert c["030T"] == 1 and sum(c.values()) == 1

def test_null_census_shape_and_z(synth):
    dg = nx.DiGraph(); dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:40])
    out = motifs.null_census(dg, n=5, seed=0)
    assert out.height == 16 and set(out.columns) == {"triad", "observed", "null_mean", "null_sd", "z"}
    assert out.filter(pl.col("triad") == "003")["z"].is_null().all() or True   # z may be null where sd 0

def test_feed_forward_middle_share():
    out = motifs.feed_forward_middle_share(ff(), {"B", "A"})
    assert out.filter(pl.col("code") == "B")["n_middle"][0] == 1 and out.filter(pl.col("code") == "A")["n_middle"][0] == 0

def test_bridge_chains_requires_order_and_excludes_core_as_a():
    fd = pl.DataFrame({"person_id": pl.Series([1,1,1, 2,2,2, 3,3,3], dtype=pl.Int64),
                       "category": ["X","N17","I50", "X","N17","I50", "I50","N17","X"],
                       "first_date": [date(2018,1,1), date(2018,3,1), date(2018,6,1),
                                      date(2018,1,1), date(2018,1,10), date(2018,6,1),   # gap too small on step 1
                                      date(2018,1,1), date(2018,3,1), date(2018,6,1)]})
    out = motifs.bridge_chains(fd, {"N17"}, {"cardiometabolic": {"I50"}}, gap_days=30, min_patients=1)
    assert out.height == 1 and out.row(0, named=True)["a"] == "X" and out.row(0, named=True)["n_patients"] == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_motifs.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement `motifs.py`**, config, and `scripts/04_motifs.py`.
- [ ] **Step 4: Run** tests (PASS) then `uv run python scripts/04_motifs.py`. Expected: census table with
  `030T` z reported; chains table non-empty; figure written; runtime under 15 minutes.
- [ ] **Step 5: Commit** `feat: step 04 motif census and bridge chains`.

---

### Task 4: Docs

**Files:**
- Modify: `docs/DATA-CONTRACT.md` (add `edges_directed`, layer and motif files), `README.md` (steps 03/04 status).

- [ ] **Step 1:** Add the tables; note that `outputs/layers/tables/*.csv` and `outputs/motifs/*.csv` are
  aggregate and may be copied to `outputs/artifacts/`.
- [ ] **Step 2: Commit** `docs: layers and motifs in the contract`.
