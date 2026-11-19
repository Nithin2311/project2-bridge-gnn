# Plan 6: Report repo, number macros, draft, supplement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A separate LaTeX repo `project2-bridge-gnn-report` that builds the 6-8 page seven-section report and a
supplement from synced pipeline artifacts, with every result number injected as a generated macro.

**Architecture:** Mirror of `project1-bridge-dx-report` (compact two-column `article`, `make`, `sync_from_pipeline.sh`).
A generator in the pipeline repo (`scripts/report_macros.py`) turns the artifact CSVs into `macros.tex`; the sync script
copies it with the tables, figures and Dylan's fragments. Prose lives in `sections/*.tex` and references only macros,
tables and figures. Runs in parallel with the fresh-clone re-run (Plan 5 Task 6); re-synced from the re-run's artifacts
before the panel.

**Tech Stack:** pdflatex + bibtex (BasicTeX, no latexmk), polars, pytest, gh CLI.

**Spec:** `docs/CLOSING-DESIGN.md` section 5.3 (and 2, 9).

## Global Constraints

- Rubric: seven sections in order (Introduction; Related Works; Graph Construction; Model Details; Results;
  Experimental Analysis; Interpretation and Conclusion), 6-8 pages double-column (~3,000 words), 5+ citations,
  every figure/table captioned and referenced, leakage statement, limitations, recorded-care (not causal) framing.
- Page rule: main PDF with references 6-8 pages (target 7); appendix content in `supplement.pdf`.
- No hand-typed result numbers: any three-decimal metric or any count that comes from an artifact is a macro.
- No patient-level data in the report repo (figures and tables are category/run-level).
- Dylan's fragments (`report/*.tex`, `dylan-refs.bib`) are used as written, edited only for flow; he is co-author.
- Template: Project 1's `main.tex` preamble (10 pt, twocolumn, 0.7 in margins, `ieeetr` bibliography style).

## Review Focus

1. A macro referenced in prose but not generated → LaTeX "Undefined control sequence". (Task 3 step: the build fails on
   `\res` with an unknown key because the lookup macro raises an error, not an empty string.)
2. A pipeline re-run changes a run id's seed set → `report_macros.py` must fail loudly on `n_seeds != 5` for main runs,
   not emit a stale value. (Task 2 test `test_macros_require_five_seeds_for_main_runs`.)
3. Underscore/percent in generated macro text (`cancer_hematologic`, `25%`) → LaTeX error. (Task 2 test
   `test_macros_escape_latex_specials`.)
4. Page count drifts past 8 after a sync. (Task 4 step: `pdfinfo` page check in the Makefile `check` target.)
5. Sync from the wrong tree (main checkout vs re-run clone) → the sync script prints the source path and the
   reference commit it reads from `REPRODUCTION.md` when present. (Task 1 step 5.)

---

### Task 1: Scaffold the report repo

**Files (new repo `../project2-bridge-gnn-report/`):**
- Create: `main.tex`, `supplement.tex`, `macros.tex` (placeholder, overwritten by sync), `sections/01-introduction.tex` ...
  `sections/07-conclusion.tex` (headings only), `refs.bib`, `Makefile`, `sync_from_pipeline.sh`, `README.md`, `.gitignore`,
  `figures/.gitkeep`, `tables/.gitkeep`, `fragments/.gitkeep`

- [ ] **Step 1**: `mkdir ../project2-bridge-gnn-report && cd` there; `git init -b main`; copy `../project1-bridge-dx-report/Makefile`
  and extend it: `all` builds `main` and `supplement`; `check` target runs `pdfinfo main.pdf | grep Pages` and fails unless 6-8;
  `clean` removes aux files for both.
- [ ] **Step 2**: `main.tex` = Project 1 preamble (copy lines 5-17 of its `main.tex`), `\input{macros}`, title "Do Bridge
  Diagnoses and Graph Structure Help Predict Disease-Group Entry? Graph Neural Networks on an EHRShot Comorbidity Network",
  authors Nithin Palyam and Dylan Girrens, date line "CAI 5155 ... Project 2, Fall 2026", abstract placeholder,
  `\input{sections/0N-...}` for N=1..7, `\bibliographystyle{ieeetr}\bibliography{refs}`.
  `supplement.tex` = same preamble, title "Supplement", sections: Selected configurations; Bootstrap intervals;
  Secondary target; Strata; Reporting checklist; inputs `fragments/baseline-selected-configs.tex`, `fragments/baseline-ci-rows.tex` (wrapped in the table skeleton from `report/baseline-ci-table.tex`).
- [ ] **Step 3**: `refs.bib` = `report/dylan-refs.bib` (7 entries) + the Project 1 entries the text will cite
  (`wornow2023`, `kapoor2023`, `hidalgo2009`, `crisafulli2026`, `singh2026`, `traag2019`; copy from `project1-bridge-dx-report/refs.bib`, dedupe keys).
- [ ] **Step 4**: `.gitignore`: `*.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz`; PDFs tracked (so the panel can read them from the repo).
- [ ] **Step 5**: `sync_from_pipeline.sh [SRC=../project2-bridge-gnn]`: copies `SRC/outputs/artifacts/eval/figures/*.pdf`,
  `SRC/outputs/artifacts/motifs/fig_motif_z.pdf` → `figures/`; `SRC/outputs/artifacts/eval/tables/*.tex` → `tables/`;
  `SRC/report/*.tex` → `fragments/`; `SRC/report/macros.tex` → `macros.tex`; prints the source path and, if
  `SRC/outputs/artifacts/REPRODUCTION.md` exists, its first line (which names the re-run commit). `set -euo pipefail`.
- [ ] **Step 6**: `README.md` modelled on Project 1's (build, sync, layout, Overleaf note: "link via GitHub sync; no class files to move").
- [ ] **Step 7**: `make` → both PDFs build with placeholder content; `git add -A && git commit -m "Scaffold report repo (Project 1 template)"`.
- [ ] **Step 8**: `gh repo create Nithin2311/project2-bridge-gnn-report --private --source . --push`; add Dylan as collaborator
  with `gh api -X PUT repos/Nithin2311/project2-bridge-gnn-report/collaborators/<dylan-login> -f permission=push` (login from
  the pipeline repo's collaborator list: `gh api repos/Nithin2311/project2-bridge-gnn/collaborators --jq '.[].login'`).

---

### Task 2: Number macros generator (pipeline repo)

**Files:**
- Create: `src/bridgegnn/report_macros.py`, `scripts/report_macros.py`
- Test: `tests/test_report_macros.py`

**Interfaces:**
- Produces: `report_macros.build(artifacts: Path, rows_summary: dict | None) -> str` returning the `macros.tex` text:
  1. `\makeatletter\newcommand{\res}[3]{\@ifundefined{res:#1:#2:#3}{\PackageError{macros}{unknown key #1/#2/#3}{}}{\csname res:#1:#2:#3\endcsname}}\makeatother`
     and one `\expandafter\def\csname res:<run_id>:<target>:<col>\endcsname{<value>}` per (run_id, target) in `eval/tables/runs_agg.csv`
     for cols `auroc_mean auroc_sd auprc_mean auprc_sd brier_mean auroc_seed auroc_lo auroc_hi auprc_seed auprc_lo auprc_hi n pos`
     (floats `%.3f`, ints as ints);
  2. `\pair{run}{baseline}{target}{col}` the same way from `paired.csv` for `d_auroc d_auroc_lo d_auroc_hi d_auprc d_auprc_lo d_auprc_hi` (signed `%+.3f`);
  3. scalar `\newcommand`s: from `layers/layer_stats.csv` (`\nNodesU \nEdgesU \nEdgesD \nEdgesBH \nEdgesBB`), `motifs/triad_census.csv`
     (`\motifZ \motifZu \motifPu` for triad `030T`), `pre2020/summary.csv` (`\nEdgesPre \preJaccard \nDevPre`), `results/gnn.csv`
     (`\nGnnRuns` = distinct run ids, `\nSeeds`), and `rows_summary` when given (`\nRows \nDevPatients \nTestPatients \nPosCM \nPosHeme \nPosMH \nPosMeta`,
     computed by the CLI from `outputs/rows.parquet`, never written to disk as rows).
  Values pass through `tex_escape` (`_ % & #` escaped) before emission. Raises `ValueError` if any run in
  `{G-gcn,G-sage,G-gat,B-prevalence,B-lr,B-hgb,B-hgb_all,B-mlp,B-mlp_pool,B-bridge_lr}` has `n_seeds != 5` on a primary target.
- CLI: `uv run python scripts/report_macros.py [--artifacts outputs/artifacts] [--rows outputs/rows.parquet] [--out report/macros.tex]`.

- [ ] **Step 1: Write the failing tests** (`tests/test_report_macros.py`; build a tiny artifacts tree in `tmp_path` with the CSV headers above)

```python
def test_macros_emit_res_and_pair_entries(tmp_path): ...  # contains r"\csname res:G-sage:cardiometabolic:auroc_mean\endcsname{0.612}" and a \pair entry with "+0.080"
def test_macros_require_five_seeds_for_main_runs(tmp_path): ...  # n_seeds 4 for G-sage/cardiometabolic → ValueError
def test_macros_escape_latex_specials(): assert report_macros.tex_escape("a_b 25% #1") == r"a\_b 25\% \#1"
def test_macros_scalars_from_layer_stats(tmp_path): ...  # r"\newcommand{\nEdgesU}{9115}" present
```

- [ ] **Step 2: Run** → FAIL (module missing).
- [ ] **Step 3: Implement** module and CLI.
- [ ] **Step 4: Run tests** → PASS; `uv run ruff check .`.
- [ ] **Step 5: Generate on the current artifacts**: `uv run python scripts/report_macros.py` → `report/macros.tex`; `grep -c "csname res:" report/macros.tex` ≥ 500.
  Add `report/macros.tex` to the `08b` label of `scripts/run_all.sh` (after `08_report_baselines.py`).
- [ ] **Step 6: Commit**: `report: generate macros.tex from the artifact tables (every number in the report is a macro)`; push.

---

### Task 3: Sections 1-4 (Introduction, Related Works, Graph Construction, Model Details)

**Files (report repo):** `sections/01-introduction.tex`, `02-related.tex`, `03-graph.tex`, `04-models.tex`, `main.tex` (abstract)

- [ ] **Step 1**: `./sync_from_pipeline.sh` (current artifacts; re-synced later from the re-run).
- [ ] **Step 2: Introduction** (~350 words): clinical motivation (multimorbidity, care across specialties), Project 1 in two
  sentences (\nNodesU nodes, \nEdgesU edges, stable cores, 13 bridge survivors), the question ("does the comorbidity graph, and
  do its bridge diagnoses, improve 12-month prediction of first entry into a stable core beyond a flat model?"), the task in
  one sentence (landmark rows, \nRows rows, canonical split), contributions as four bullets (protocol, directed/null/time-respecting
  layers, bridge ablations, honest negative).
- [ ] **Step 3: Related Works** = `\input{fragments/related-work}` plus one closing paragraph positioning against Project 1's
  lineage (Hidalgo; Kapoor leakage) — keep Dylan's five citations and add at most two.
- [ ] **Step 4: Graph Construction** (~550 words): cohort and preprocessing (from `docs/DESIGN.md` §3-4: era, lookback 1,095 d,
  washout 24 mo, horizon 12 mo, censoring), nodes/edges (Layer U from Project 1; Layer D: \nEdgesD directed, binomial test, gap 30 d,
  day-one codes dropped, a DAG; U-bh \nEdgesBH over the 190,036-pair family; U-bb; five rewired + one shuffled null), node features by
  group (CODE DEMO UTIL TEMP STRUCT BRIDGE; cite `feature_groups.json`), target masking (core ∪ chapter block), the leakage statement
  (test patients never in the graph; selection inside development only; input masks; the 5a time-level contamination accepted and
  tested by the pre-2020 rebuild: \nEdgesPre edges, Jaccard \preJaccard, cardiometabolic core empty before 2020), motif census in two
  sentences (\motifZ free vs \motifZu restricted). Table: `tables/layer_stats` if generated, else a 5-row hand table of layer sizes using macros.
- [ ] **Step 5: Model Details** (~500 words): per-row subgraph classification; GCN/SAGE/GAT, depth 2, hidden 64, mean readout; CODE as
  learnable embedding; loss BCE with pos_weight and prior correction at prediction; protocol (5-fold GroupKFold on development, grid
  lr x weight decay, early stopping on fold AUPRC, refit for mean best epoch, test once, seeds 0-4; note GCN won dev AUPRC on heme, SAGE elsewhere,
  SAGE kept as the default ablation backbone); then `\input{fragments/baseline-methods}`.
- [ ] **Step 6: Abstract** (≤200 words) with macros for the headline numbers.
- [ ] **Step 7**: `make` builds; `grep -E "[0-9]\.[0-9]{3}" sections/0[1-4]*.tex` returns nothing (all metrics are macros);
  `grep -i "undefined" main.log` returns nothing. Commit `Draft sections 1-4`.

---

### Task 4: Sections 5-7, supplement, page budget

**Files (report repo):** `sections/05-results.tex`, `06-experiments.tex`, `07-conclusion.tex`, `supplement.tex`, `Makefile` (check)

- [ ] **Step 1: Results** (~500 words): main table = `fragments/baseline-results` (Dylan's table* with `baseline-rows`), one paragraph
  per target using `\res`; paired deltas paragraph from `fragments/baseline-results-summary` (G-sage vs B-hgb_all / B-mlp_pool) plus the
  sentence "all twelve intervals include zero" written as a macro-checked statement (`\pair` values quoted); calibration (Brier at the
  prevalence floor, slope/CITL from `tables/runs_agg`), figures: `roc_pr_cancer_hematologic.pdf` and `reliability_cancer_hematologic.pdf`
  (heme as the best-powered target) side by side; class balance (\nPosCM \nPosHeme \nPosMH, test n and pos from `\res{...}{n}`).
- [ ] **Step 2: Experimental Analysis** (~700 words), one subsection per family, each with one table or figure and a paired-interval
  sentence where Task 1 of Plan 5 added a pairing: A (A1a/A1b vs G-sage with `\pair`; A2a shows covariates matter), B (B1 directed,
  B3/B4/B6 within noise; B5/BT with `\pair{B5-P}{B5-U}` and `\pair{BT-P}{BT-U}`), C (`\input{fragments/experiments-c1-d1}` C1 part),
  D (null graphs: `fragments/null-summary` and `\pair{D1-shuffle}{G-sage}`), E (depth/heads/hidden/aggregator table built from `\res` rows,
  mean±SD, six rows).
- [ ] **Step 3: Interpretation and Conclusion** (~450 words): findings (GNN ≈ best flat model; bridges a compact informative summary;
  structure adds nothing measurable; mental health at chance), recorded-care framing, what worked / did not, limitations (positives per
  target, single centre, calibration slopes, time-level contamination and the empty pre-2020 cardiometabolic core, five null graphs, no
  external validation), future work (burden-matched null, second dataset, bipartite patient-code graph: "designed, not run" unless the
  extra credit happens).
- [ ] **Step 4: Supplement**: selected configurations (`fragments/baseline-selected-configs`), the CI table (`fragments/baseline-ci-table`
  skeleton + `baseline-ci-rows`), secondary target table from `\res{...}{cancer_metastatic}{...}` for G-* and the four baselines with the
  footnote from the design (L1), strata summary from `tables/strata.csv` (a short hand table of AUROC by sex / age band for G-sage, macros
  via a `\strat` lookup added to the generator if needed, else quoted from the CSV in the table caption's source note), TRIPOD+AI-style
  checklist (items → section numbers).
- [ ] **Step 5**: `make && make check` → main 6-8 pages; if over 8, move the E table or the strata paragraph to the supplement; if under 6,
  bring the CI table back.
- [ ] **Step 6**: Reference check: every `\ref`/`\cite` resolves (`grep -c "Warning: (Citation|Reference)" main.log` → 0); every table and
  figure is referenced in text (`for l in $(grep -oh "\\label{[^}]*}" sections/*.tex fragments/*.tex | sed 's/\\label{//;s/}//'); do grep -q "ref{$l}" sections/*.tex || echo "unreferenced $l"; done`).
- [ ] **Step 7**: Three-decimal grep on sections 5-7 returns nothing; commit `Draft sections 5-7 and supplement`; push.

---

### Task 5 (after Plan 5 Task 7): re-sync from the re-run, freeze for the panel

- [ ] **Step 1**: In the pipeline main checkout, after Plan 5 Task 7 replaced `outputs/artifacts/` with the re-run's, regenerate the
  fragments and macros from those artifacts (fragments live in `report/`, which the artifact copy does not touch):
  `uv run python scripts/08_report_baselines.py --runs outputs/artifacts/eval/tables/runs_agg.csv --paired outputs/artifacts/eval/tables/paired.csv --baseline-results outputs/artifacts/results/baselines.csv`
  and `uv run python scripts/report_macros.py --artifacts outputs/artifacts`; commit `report: fragments and macros from the re-run artifacts`.
- [ ] **Step 2**: Report repo: `./sync_from_pipeline.sh && make && make check`; read the PDF once for numbers that moved; commit
  `Sync figures, tables, macros from the fresh-clone re-run (<hash>)`; tag `panel-2026-11-25`; push with tags.
- [ ] **Step 3**: Hand the panel: `main.pdf`, `supplement.pdf`, the report repo at the tag, the pipeline repo at the re-run commit,
  `outputs/artifacts/` incl. `REPRODUCTION.md`, the rubric PDF, `docs/DESIGN.md`, `docs/CLOSING-DESIGN.md` §5.4 (reader briefs).
