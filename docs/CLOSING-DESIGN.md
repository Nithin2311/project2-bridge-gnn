# Project 2 closing phase: leftovers, fresh-clone re-run, report, independent review, extra credit

Status: DRAFT for approval, 2026-11-19. Follows `docs/DESIGN.md` (the project design) and closes it out.
Authors: Nithin Palyam. Dylan Girrens contributes report sections per his fragments.
Deadline: 2026-12-04 (submit by Dec 3).

Items marked **[default]** are working defaults; override freely. Section 9 lists the calls made after the
builder check of 2026-11-19.

---

## 1. Intent and why

Everything the design asked for is built and merged: steps 00-09, the main GNN comparison, the five rubric
experiment families, the time-respecting rebuild and backtest, Dylan's baselines, step 08 tables and figures.
The findings are settled: GNNs match but do not beat a tuned flat model; bridge summaries are a compact,
informative feature set; message passing over the comorbidity graph adds nothing measurable.

Three things stand between that and a submission we would defend in front of Dr. Kim's collaborator:

1. **Proof that it all reproduces** from a fresh clone and the README, not from a checkout with months of state.
2. **A report** that states the claims at the strength the numbers support, in the rubric's seven sections.
3. **An independent review** of that report and the code behind it, with the findings fixed before submission.

Order, as decided on 2026-11-19: close the leftovers, launch the fresh-clone re-run, draft the report while it
runs, convene the review on the frozen PDF and artifacts, fix, rebuild, then extra credit if time remains.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. Every number in the submitted report comes from a fresh-clone run of the committed code, and every
  aggregate table matches the previously committed artifacts (exactly for deterministic steps, within seed
  noise for torch).
- G2. A 6-8 page, seven-section report in a separate LaTeX repo, built with `make`, numbers injected from the
  synced tables, Dylan's fragments integrated, 5+ relevant references, leakage statement, limitations.
- G3. A five-reader independent panel on the compiled PDF plus repo, artifacts and re-run log; every
  Critical and Important finding fixed or explicitly declined with a reason; affected pieces re-run.
- G4. Leftovers closed: secondary target runs, step 08 runtime, a one-command `run_all.sh`, the stale branch.
- G5. Extra credit (B2 bipartite HeteroGNN, rubric Option A) only if at least three working days remain after
  the review fixes; otherwise explicitly not attempted.

**Non-goals**
- New experiments beyond the design. The grid is done.
- Changing the task, the split, the targets or the primary metrics.
- Spring publication work (burden-matched null, second dataset).

**Anti-goals**
- A report sentence stronger than its interval supports.
- Any number in the PDF typed by hand rather than generated.
- A re-run that "passes" by reusing cached outputs.
- Patient-level data reaching the report repo or GitHub.

## 3. Constraints

- Two people; Nithin on everything below except Dylan's report sections; Dylan's availability unknown
  (graduating).
- Compute: M4, 24 GB, CPU. Full pipeline from scratch about 9 hours (step 01 4 min, 03 10 s, 04 8 min,
  main comparison 11 min, grid about 6 h, baselines about 1.5 h, step 08 about 1.5 h today, step 09 11 min).
- The EHRShot CSVs (~13 GB) stay where they are; the fresh clone points at them through `EHRSHOT_DATA`.
- No `latexmk`; `pdflatex` + `bibtex` via the Project 1 Makefile.

## 4. Approaches considered

| Decision | Options | Picked | Why |
|---|---|---|---|
| Re-run | (a) wipe `outputs/` in the current checkout; (b) fresh clone, fresh venv, README only | (b) | Only (b) proves "runnable code"; (a) can pass on leftover state |
| Grid in the re-run | minimum set vs full 87 runs | full | Compute is free overnight; partial re-run leaves unverified numbers in the report |
| Report location | `report/` in the pipeline repo vs separate repo like Project 1 | separate repo | Project 1 precedent, Overleaf GitHub sync, clean separation of code and prose |
| Review format | ad hoc agents vs the Project 1 panel (EIC + 3 referees + devil's advocate, independent) | panel | Worked on Project 1; independence between readers is the point |
| Extra credit | commit now vs conditional on slack | conditional | Review fixes come first; a half-done bonus model costs more than it earns |

## 5. The parts, as decided

### 5.1 Leftovers (before the re-run launches)

| Item | What | Acceptance |
|---|---|---|
| L1 | `cancer_metastatic` secondary target: G-gcn/sage/gat plus `B-prevalence`, `B-hgb_all`, `B-mlp_pool`, `B-bridge_lr` (5 seeds each), via `experiments.yaml` `targets:` overrides, no code change; paired G-sage vs B-hgb_all/B-mlp_pool through the same step 08 rule as the primary targets | 35 results rows; appendix table with a footnote: 117 positives, same selection protocol (hyperparameters chosen on its own development rows by GroupKFold; "secondary" means no design decision was taken on it), underpowered |
| L2 | Step 08 runtime: paired bootstraps only for `eval.baselines_for_pairing` against runs listed in a new `eval.paired_gnn_runs` (default `[G-gcn, G-sage, G-gat]`) plus `eval.paired_runs`, which gains the ablation headlines `[A1a, G-sage]`, `[A1b, G-sage]`, `[B1, G-sage]`, `[D1-shuffle, G-sage]`, `[D1-rewire0, G-sage]` so the bridge, directed-edge and null-graph sentences in section 6 rest on identical-row intervals, not seed SD alone; everything else skipped | step 08 under 15 min on the full predictions folder |
| L3 | `scripts/run_all.sh`: steps 00, 01, 02, 03, 04, 05 (all models, primary targets, via `07 --only-baselines`), 06 main comparison (G-* via `07 --ids G-gcn G-sage G-gat`), grid (`07` remaining ids), 09, 08, `08_report_baselines`; logs each step with timestamps to `outputs/logs/run_all.log`; stops on first failure; `--quick` smoke variant | one command reproduces every artifact |
| L4 | Step 07 driver runs the GNN grid (replaces the shell loop used on Nov 9); subsample files for C1 come from `write_subsamples` (no-overwrite); `experiments.yaml` gets `G-*` for `cancer_metastatic` as a `targets:` override | `07 --dry-run` lists 29 experiments x 3 targets + 3 x metastatic |
| L5 | Stale remote branch `dylan/project2-deliverables` deleted (the branch-delete push was blocked by the tool permission layer; Nithin does it on GitHub). Ordinary pushes to `main` worked throughout (Nov 9, 12, 13) and are assumed to keep working; if one is blocked, the fresh clone is taken from the local repository at a recorded commit (identical tree) and `REPRODUCTION.md` records the commit and the GitHub URL | branch gone |
| L6 | README "Reproduce from scratch" section: clone, `uv sync --extra dev --extra gnn`, `export EHRSHOT_DATA`, `./scripts/run_all.sh`, expected runtime, how to compare with `outputs/artifacts/` | a reader can follow it |

### 5.2 Fresh-clone re-run

- Clone `https://github.com/Nithin2311/project2-bridge-gnn.git` into a sibling folder
  `project2-bridge-gnn-rerun/` (so `../project1-bridge-dx` and `../FinalProject/data` resolve as in the
  README); `uv sync --extra dev --extra gnn`; `uv run pytest -q`; `./scripts/run_all.sh` in the background with
  the log kept.
- Project 1 is a path dependency; its `outputs/` must be the Oct 23 run (it is). Project 1 itself is not re-run
  (frozen by design; its own reproducibility was checked in October).
- Comparison script `scripts/compare_runs.py OLD NEW`: for every CSV under `outputs/artifacts/results/`,
  `outputs/eval/tables/`, `outputs/layers/tables/`, `outputs/motifs/`, `outputs/pre2020/summary.csv` and the
  node matrix: report exact match, or max absolute difference per numeric column; thresholds: deterministic
  steps (01-05, 09 rebuild, baselines) must match exactly; torch runs (06) within 0.02 AUROC per (run, target,
  seed) and identical selected hyperparameters **[default]**; anything outside is a finding for the review.
- The re-run's `outputs/` becomes the source of truth: its artifacts replace `outputs/artifacts/` on `main`,
  with the comparison report committed beside them (`outputs/artifacts/REPRODUCTION.md`).

### 5.3 Report repo `project2-bridge-gnn-report`

- Layout mirrors Project 1: `main.tex`, `sections/01..07.tex`, `refs.bib`, `figures/`, `tables/`, `Makefile`
  (copied), `sync_from_pipeline.sh` (copies `outputs/artifacts/eval/tables/*.tex`,
  figures, Dylan's `report/*.tex` fragments and `dylan-refs.bib`; rewrites wide tables to `table*`), `README.md`.
  Private GitHub repo under Nithin's account, Dylan as collaborator; Overleaf linked to it if wanted.
- Sections per the rubric: 1 Introduction (clinical motivation, link to Project 1 findings, question, target,
  contributions); 2 Related Works (Dylan's five plus Project 1's lineage papers as needed, 5+ total); 3 Graph
  Construction (cohort, preprocessing, nodes, edges, layers U/D/U-bh/U-bb/N, statistics, leakage statement
  incl. the 5a paragraph and the pre-2020 rebuild); 4 Model Details (task, GCN/SAGE/GAT, baselines incl. Dylan's
  methods fragment, split, seeds, loss, grid, early stopping, prior correction); 5 Results (main table with
  intervals, calibration, curves, strata, class balance, paired deltas); 6 Experimental Analysis (A-E with one
  table or figure each, D1 and C1 from Dylan's fragment, B5/backtest, motif census); 7 Interpretation and
  Conclusion (findings, recorded-care framing, what worked, limitations incl. positives per target, calibration,
  single centre, time-level contamination, mental-health at chance; future work incl. spring items). Appendix:
  selected configurations, TRIPOD+AI-style checklist, extra tables.
- Numbers: every figure in the text is a macro in `macros.tex` generated by `scripts/report_macros.py` in the
  pipeline repo from the artifact tables (counts, AUROCs, intervals, deltas); the sync script copies it. No
  hand-typed result numbers **[default]**.
- Template: Project 1's compact two-column `article` class (10 pt, 0.7 in margins), which is what Project 1 actually
  shipped; the vendored IEEEtran is not used. The rubric asks only for "6-8 pages in double-column (~3,000 words)".
- Page rule **[default]**: the main PDF, references included, is at most 8 pages and at least 6; target 7. The
  appendix (selected configurations, CI tables, metastatic table, TRIPOD+AI checklist) is a separate
  `supplement.pdf` built from the same repo and submitted beside the report, so the rubric's limit is met under the
  strictest reading. If the main text lands under 7 pages, the CI table moves back into it.
- Writing split: Nithin drafts 1, 3, 4 (GNN part), 5, 6 (A, B, D1 narrative beyond Dylan's, E, B5), 7; Dylan's
  fragments fill 2, 4 (baselines), 6 (C1, D1 tables). Dylan reviews the whole draft before the panel if
  available; otherwise the panel reviews without him.

### 5.4 Independent review panel

- Convened once the re-run comparison is clean and the PDF builds from the re-run's artifacts.
- Five fresh readers, run independently, none sees another's report: Editor-in-chief (rubric fit, claim strength,
  completeness, page limit); Referee 1 methods and statistics (leakage, protocol, intervals, calibration,
  multiplicity, the 5a argument); Referee 2 clinical informatics (clinical framing, recorded-care language,
  target definitions, interpretability of bridges, limitations); Referee 3 graph ML and reproducibility
  (architectures, ablation design, null models, code vs text agreement, fresh-clone log, artifact completeness);
  Devil's advocate (the strongest case that the headline is wrong or trivial).
- Each receives: the PDF, the report source, the pipeline repo at the frozen commit, `outputs/artifacts/`,
  `REPRODUCTION.md`, the rubric PDF, `docs/DESIGN.md`. Each returns Strengths / Critical / Important / Minor /
  Declined with file and line references, like the Project 1 panel.
- Synthesis by Nithin: one `REVIEW-PANEL-<date>.md` in the report repo listing every finding, its disposition
  (fixed, declined with reason, deferred to spring), and the commits that address it. Critical and Important
  findings are fixed; if a fix changes code, the affected steps are re-run in the rerun clone and the
  comparison refreshed; the PDF is rebuilt. Minor findings are fixed if under 15 minutes each, else listed.
- A second, lighter pass (editor only) reads the revised PDF for regressions.

### 5.5 Extra credit, conditional

- Trigger: all panel fixes merged and at least three working days before Dec 3.
- Scope as `docs/DESIGN.md` section 6 B2: per-row heterogeneous graph with one patient node (DEMO + UTIL)
  linked to its code nodes, code-code edges from Layer U, PyG `HeteroConv` over SAGE layers, readout from the
  patient node; same protocol, same seeds; compared against the three GNNs and the baselines in the same tables;
  cited (heterogeneous GNN and patient-code bipartite precedents from the existing literature briefs). Reported
  honestly even if it does not improve.
- If not triggered: the report says in one sentence that the bipartite variant was designed but not run.

## 6. Timeline

| Dates | Work |
|---|---|
| Nov 19-20 | L1-L6; push; fresh clone; launch `run_all.sh` (overnight) |
| Nov 20-24 | Report repo scaffold, macros generator, sections 1-7 drafted; sync from the re-run as artifacts land; Dylan's fragments integrated |
| Nov 24 | Comparison script on the finished re-run; `REPRODUCTION.md`; artifacts replaced on `main`; PDF built from re-run numbers |
| Nov 25 | Panel (five readers) |
| Nov 26-29 | Fix pass, targeted re-runs, PDF rebuild; editor re-read |
| Nov 30-Dec 1 | Extra credit if triggered |
| Dec 2-3 | Final build, submission package (report PDF, supplement PDF, repo link or zip, artifacts), submit |

## 7. Risks

| Risk | Response |
|---|---|
| Fresh-clone run fails partway (environment, path) | That is the point; fix the README or code, restart from the failed step (steps are idempotent and write to `outputs/`) |
| Torch results drift beyond 0.02 between machines or runs | Report the drift in `REPRODUCTION.md`; the submitted numbers are the re-run's; if drift changes a conclusion, say so |
| Panel finds a methods flaw that invalidates a family of results | Fix, re-run that family only (minutes to an hour each), rebuild; the report's claims are already cautious |
| Dylan unavailable | His fragments are complete; Nithin integrates and takes responsibility for the whole text; authorship unchanged |
| Page limit | Supplement absorbs selected configurations, CI tables, metastatic and extra experiments; references stay inside the 8 |
| Time | Extra credit is the only droppable item and is last by design |

## 8. Left to the builder

Exact macro names; LaTeX styling within the Project 1 template; comparison-script tolerances per column beyond
the stated defaults; which figures make the main text versus the appendix; `run_all.sh` flag names.

## 9. Calls made after the builder check (2026-11-19)

| Question raised | Call |
|---|---|
| Which template, and do references and appendix count toward 6-8 pages? | Project 1's `article` template; main PDF with references at most 8 pages; appendix as a separate supplement (5.3) |
| Paired intervals for the ablation headlines, or seed SD only? | Paired: A1a, A1b, B1, D1-shuffle, D1-rewire0 against G-sage added to `eval.paired_runs` (L2) |
| Metastatic: GNN-only rows, and the "never tuned on" wording | Add prevalence, all-code boosting, pooled MLP and bridge LR on metastatic; footnote states the actual protocol (L1) |
| Pushes on the critical path | Pushes to `main` have worked all along; fallback is a local clone at a recorded commit (L5) |

Verified by the builder check, so not re-planned: GNN code unchanged since the Nov 9 grid; `outputs/features/node_x.npy`
equals the cross-check matrix the runs used; `outputs/eval/` equals `outputs/artifacts/eval/`; all 36 GNN and 9 baseline
run ids carry 15 rows; the Project 1 panel ran five fresh readers in series.
