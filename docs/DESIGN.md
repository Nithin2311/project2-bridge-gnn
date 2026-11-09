# Project 2 design: do bridge diagnoses and graph structure predict entry into a disease community?

Status: DRAFT v3 for approval, 2026-11-06. v1 (Nov 5) independently reviewed; v2 resolved its five review questions; v3 folds in the Project 2 description (`Project Description_Project 2.pdf`, received Nov 6).
Plan 1 (foundation) is implemented and merged as of 2026-11-09.
Authors: Nithin Palyam (plan), Dylan Girrens (to review and take ownership of blocks in section 10).
Inputs: Project 1 pipeline and outputs, P2-READINESS.md, the two P2 literature briefs, the Project 2 description
and rubric, Dr. Kim's scoping call of 2026-11-05.

Items marked **[default]** are working defaults, not decisions anyone has made; override freely.
Items marked **[open]** need a human answer before implementation. Items marked **[rubric]** are required by the
Project 2 description.

---

## 1. Intent and why

Project 1 built a statistically validated ICD-10 comorbidity network on EHRShot's train+val patients and
screened 15 "bridge" diagnoses between four target cores. Its own analysis showed the bridges are mostly
high-degree complication codes that may mark diagnostic burden rather than shared disease. Project 2 turns
that into a testable question on patients the graph has never seen:

> For a patient who has not yet received any diagnosis from a target community, does the structure of their
> existing diagnoses in the comorbidity graph, and specifically the presence of bridge diagnoses, predict
> that they will enter that community within 12 months, beyond what a flat list of their codes predicts?

Three things this serves at once:

1. The course deliverable (due 2026-12-04): a leakage-free graph ML experiment on the Project 1 graph; GCN,
   GraphSAGE, GAT plus a non-graph baseline on one task, split and feature set; five families of controlled
   experiments; a 6-8 page report. **[rubric]**
2. Dr. Kim's publication direction: bridge diagnoses used to predict future diagnoses; temporal, directed
   edges; small directed subgraph (motif) frequencies; demographic comparisons; EHRShot first.
3. An honest answer. A well-run negative result ("structure adds nothing over bag-of-codes at this sample
   size") is acceptable for the course (the rubric asks "whether graph structure adds value beyond the
   non-graph baseline") and useful for the paper. The design must make that answer credible.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. One prediction task, fully specified before any model is trained: landmark rows, 12-month horizon,
  stable-core target definition, canonical split, censoring rule. (All already built in P1 step 9.)
- G2. A fair ladder of models on identical rows and splits: trivial baseline, bag-of-codes linear model,
  gradient-boosted trees, MLP on the same features the GNN sees, then GCN, GraphSAGE, GAT on patient
  subgraphs.
- G3. One controlled experiment in each rubric family (feature, graph structure, scale and subgroup, null
  model, model), plus the ablations that answer the research question: bridge features in/out, undirected
  co-occurrence vs directed precedence edges, lookback window. **[rubric]**
- G4. A directed temporal layer on the Project 1 graph (first-record precedence) and a 3-node directed motif
  census, as classical analysis that feeds the GNN experiments and the paper.
- G5. Results with uncertainty (5 seeds, patient-level bootstrap intervals), calibration, training curves,
  ROC and PR plots, confusion matrices, stratified by sex, age band, subgraph size and index year. **[rubric]**
- G6. Everything reproducible from the Project 1 outputs plus the 13 EHRShot CSVs by one command sequence,
  with unit tests on synthetic data; node and edge lists, feature matrices, thresholds, logs and
  hyperparameter records saved as deliverables. **[rubric]**

**Non-goals (this semester)**
- A new GNN architecture as a primary contribution. Standard PyG layers for all required models. (The
  heterogeneous bipartite variant in experiment B2 may be promoted to extra-credit Option A in week 3; see
  section 8.)
- A second dataset. TriNetX / TGH is extra-credit Option B and a spring question.
- Sequence models, temporal GNNs, survival or competing-risk heads. Noted as paper extensions.
- Rebuilding the Project 1 graph as the primary input. It is frozen; Project 2 consumes it. One
  time-respecting rebuild is run as a named experiment (B5).
- Fixing the Project 1 null-model weakness (burden-matched null). Spring work for the paper.

**Anti-goals (what would make this a failure even if it runs)**
- Any leakage into the test claim: test patients influencing the graph, node features, feature scaling,
  thresholds, or model selection; any row-level input computed from that patient's records after the index
  date; target codes or their near-synonyms present in the input. (One known population-level contamination
  of train/val rows is accepted and documented; see section 5a.)
- Reporting a GNN "win" without a tuned non-graph baseline on the same rows and features, or without seeds
  and intervals.
- Choosing the target definition, horizon, or lookback after seeing test results.
- Scope creep that leaves the course deliverable half-done on Dec 4. Section 9 fixes the minimum.

## 3. Constraints

- Deadline 2026-12-04. Four weeks, two people, Nithin with other commitments, Dylan graduating.
- Hardware: Apple M4, 24 GB. No CUDA. PyTorch MPS or CPU. Scale is small (about 10-20k subgraphs of about
  10 nodes); CPU is fine.
- Data: 13 EHRShot CSVs in `FinalProject/data/`, DuckDB cache, canonical split file. No patient-level data
  leaves the machine; no patient-level files are committed. Labs (`measurement.csv`) exist on disk but are not
  used; the rubric's "diagnosis + labs" feature ablation is marked "where available" and is skipped with that
  reason stated.
- Positives at 12 months, stable-core definition (from `p2_label_counts.csv`):

  | Target | rows | patients | positives | train / val / test positives |
  |---|---|---|---|---|
  | cardiometabolic | 1,863 | 554 | 163 | 50 / 57 / 56 |
  | cancer_hematologic | 5,903 | 1,434 | 315 | 114 / 110 / 91 |
  | mental_health | 4,819 | 1,211 | 282 | 100 / 99 / 83 |
  | cancer_metastatic | 9,705 | 2,020 | 117 | 38 / 37 / 42 |

  These are small. Test-set intervals will be wide (56 positives gives roughly plus or minus 0.07 AUROC).
  The design accepts this and says so; it does not inflate counts by changing definitions.
- Course requirements **[rubric]**, from the Project 2 description:
  - Models: GCN, GraphSAGE, GAT, and a non-graph baseline (logistic regression or MLP on the same input
    features). Same task, split, preprocessing, features. Fixed seeds. Patient-wise or canonical split.
  - Experimental analysis, five families, each with metrics, the changed factor, fixed settings, and a short
    interpretation: A node features, B graph structure, C scale and subgroup, D null models (real vs shuffled
    or degree-preserving rewired edges), E model (depth, aggregator, heads).
  - Evaluation: AUROC, AUPRC, F1 etc.; for rare outcomes prioritise AUPRC/F1 and state class balance; mean and
    SD over at least 3 runs; confidence intervals with the method stated; subgroup results by age and sex.
  - Leakage: pre-prediction records only for inputs and edges; edges, statistics and weights from training
    patients only; normalisation fit on training data only; structural features from the training graph only;
    target diagnosis and its descendant or near-synonym codes excluded from inputs.
  - Hyperparameter grid and best configuration reported: batch size, learning rate, optimiser, weight decay,
    layers, hidden size, dropout, seeds. Early stopping on a validation metric; training curves, validation
    metrics and logs retained.
  - Deliverables: node and edge lists, feature matrices, construction thresholds; documented code, README,
    dependencies, logs, hyperparameter records; checkpoints optional.
  - Report: 6-8 pages double column, about 3,000 words, seven sections (Introduction; Related Works with at
    least 5 relevant recent studies; Graph Construction including graph statistics and a leakage statement;
    Model Details; Results; Experimental Analysis; Interpretation and Conclusion).
  - Rubric weights: GNN implementation 20; Evaluation and results 15; Research question 10; Related works 10;
    Graph construction 10; Experimental analysis 10; Interpretation 10; Report and reproducibility 10; Data
    preparation 5. Extra credit up to +20% for one of: Option A own GNN variant, Option B larger benchmark.

## 4. The prediction task (frozen from Project 1 step 9)

- Unit: one row per (patient, landmark index date, target). Grid every 182 days from 2017-10-01. Eligible if
  at least 730 days of ICD-10-era history, an encounter in the prior 365 days, no target code in any history
  (era rows plus bridged pre-2015 rows), follow-up covering the horizon.
- Prediction time point = the index date. Observation window = lookback ending at the index. Outcome horizon =
  12 months. Label: first stable-core target code within (index, index + 12 months]. Censored rows (death or
  record end inside the horizon) are excluded from the primary analysis. **[rubric: all five stated
  explicitly in the report]**
- Targets: cardiometabolic, cancer_hematologic, mental_health as primary. cancer_metastatic reported as
  secondary with an explicit underpowered flag (117 positives).
- Split: canonical EHRShot train / val / test by patient. Development set = train + val (2,787 patients).
  Model selection = 5-fold GroupKFold by patient inside the development set (hyperparameters and the number of
  epochs, chosen on out-of-fold AUPRC averaged over folds). Final model = refit on the whole development set
  with the selected configuration and the mean best epoch count. Test rows are touched once per final
  configuration. **[default]** A fixed train/val protocol would fit the cardiometabolic model on 50 positives
  and select on 57; pooling gives 107 for fitting and stabilises selection. This matches the P1 config note
  ("Project 2 tunes by cross-validation inside train + val"). The report names the development set as the
  "training patients" in the rubric's sense; test patients never enter any graph, statistic, scaler or
  selection step.
- Secondary horizon 6 months, secondary target definitions (single-seed community, chapter block): appendix
  table only if time permits. **[default]** Not tuned on.
- Temporal backtest (secondary): train on rows with index <= 2019-12-31, test on held-out patients' rows with
  index >= 2020-07-01, on the pre-2020 graph (experiment B5).

## 5. Inputs the patient sees at an index date

For each row, the model may use only that patient's records dated on or before the index date.

- Code set: the patient's distinct ICD-10 categories recorded any time in a lookback window ending at the
  index. Step 01 extracts ALL categories (including R, Z, O chapters and codes below the prevalence floor), then
  derives the main-graph subset (496 nodes) for the graph models. Primary lookback: 1,095 days. Ablation: 365
  days and all history. **[default]** 1,095 is primary because 365 leaves 20% of rows with fewer than 3 codes.
  **[default]** Extracting all codes now is cheap and lets the tabular baselines see Z79, Z86, Z51 and R codes,
  which are often the strongest predictors of new diagnoses. Category mapping reuses
  `bridgedx.cohort._register_category_map` (Project 2 takes Project 1 as a path dependency) so code sets are
  identical to Project 1's.
- Target exclusion from inputs **[rubric]**: for a given target, the row's input masks every code in the
  target's stable core AND its a-priori chapter block (`chapter_blocks` in the P1 config), as the near-synonym
  set. Eligibility (no prior target code) stays defined on the core only, so label counts do not change.
  **[default]**
- Patient-level covariates: age at index, recorded sex, number of distinct categories in the window, number of
  diagnosis rows in the window (utilisation proxy), number of distinct visit days in the prior 365 days, days
  since first era diagnosis, calendar year of index.
- Temporal per-row node features **[rubric: "temporal vs no temporal"]**: for each code in the set, days since
  its most recent recording before the index and days since its first recording (log-scaled), and a flag for
  "recorded in the last 90 days".
- Static node features for each code (from `outputs/p2/node_features.csv`, computed on train+val only):
  prevalence, degree, weighted degree, betweenness and its null percentile, closeness, eigenvector, PageRank,
  participation, within-module z, degree residual z, seed-0 Leiden community id (one-hot or embedding; the file
  has no consensus id, `coassignment.csv` exists if one is wanted), stable-core membership flags per target,
  inpatient share, CCIR chronic share, mean age at first occurrence, female share, chapter one-hot, backbone
  degree.
- Bridge-specific features, pre-registered here **[default]** because the survivor set is only 13 codes and a
  13-flag ablation would be underpowered on 100 positives:
  - B-flag: binary `bridge_survivor` per node, and per target the union of survivors over the three pairs that
    involve that target (cardiometabolic 7 codes, hematologic 6, mental health 8).
  - B-score: continuous cluster-restricted betweenness per node per pair from `outputs/bridges.csv` (all 496
    nodes x 6 pairs, pre-null), joined onto the node feature matrix; per target, the node's max over the pairs
    involving that target.
  - Row features for tabular models: count of B-flag codes in the window; sum and max of B-score among the
    patient's codes toward the target.
  Experiment A1 removes both B-flag and B-score from node features (variant a), and additionally drops the
  B-flag codes from the subgraph and the bag (variant b). The report states the power limit: with 100-115
  development positives only a large effect is detectable, so a null is "not detected", not "absent".
- Feature groups, named once so every ablation refers to them: CODE (bag or code identity), DEMO (age, sex),
  UTIL (counts, visit days, days since first dx, year), TEMP (temporal per-row node features), STRUCT (P1
  structural node features), BRIDGE (B-flag, B-score, bridge row features).
- Standardisation of continuous features is fit on development rows only (and inside each fold on the
  training folds).

Leakage guards (unit-tested): no code dated after the index enters a row; no target-core or chapter-block code
appears in any row's input for that target; node features and edges derive only from train+val patients
(asserted against the split file); test patient ids never appear in any fitting, scaling, or selection call.

### 5a. Known contamination of train/val rows (decided 2026-11-05, strengthened 2026-11-06)

The Project 1 graph and node features were computed from train+val patients' full ICD-10-era records
(2015-10 to record end). Train/val rows are indexed 2017-10 to 2021-09 on those same patients, so a
development patient's own future diagnoses contributed, at population level (one patient in 2,787), to the
edges and features their rows are scored on (Kapoor and Narayanan L3.1). Test patients never touched the graph,
so test metrics are unbiased. Handling:
- The report's leakage statement describes the graph as a population prior built from the development
  cohort's full follow-up, and says in one paragraph that this is a time-level, not patient-level, exposure.
- Development-set fit and cross-validated selection may be optimistic for graph-feature models relative to
  bag-of-codes models; the test comparison is the only one quoted as a result.
- Experiment B5 (promoted from a footnote sensitivity on 2026-11-06): rebuild the graph, communities, bridge
  scores and node features from development-patient records dated before 2020-01-01 using the Project 1 code
  (path dependency, config override of the era end; about 40 minutes), then score all rows with index >=
  2020-01-01 on both graphs (P1 full-era vs pre-2020) with the same models. Reported in the Experimental
  Analysis section as a named graph-structure experiment, alongside the temporal backtest. If the result holds,
  the contamination did not drive it.
- Rolling per-era graphs as the primary input were considered and rejected for this semester (three P1 reruns,
  thinner early graphs, P1 bridge flags no longer fixed; about one extra week). Noted as spring work.

## 6. Graph construction for the models

Two edge layers over the same 496 nodes.

**Layer U (undirected, from Project 1).** The main graph: 9,115 phi-weighted edges (Fisher exact, Bonferroni,
OR >= 1.2, >= 10 shared patients). Variants: U-bb, the top-10 per-node phi backbone
(`graph_backbone_topk.graphml`, about 3,300 edges); U-bh, edges kept at BH-FDR 0.05 instead of Bonferroni
(threshold sensitivity, B4) built from the P1 pair table.

**Layer D (directed precedence, new; Dr. Kim's request).** Built from `patient_category.parquet` (person,
category, first_date) on train+val patients only.
- For each ordered pair (A, B) among main-graph nodes: n_AB = patients with first A at least g days before
  first B; n_BA the reverse; ties within g days excluded. g = 30 days **[default]**; sensitivity g = 0 and 90.
- Keep A -> B if (A, B) is an undirected edge in Layer U (direction is added to validated co-occurrence, not
  invented), n_AB + n_BA >= 20, and a two-sided binomial test of n_AB vs n_BA rejects p = 0.5 at BH-FDR 0.05.
  Weight: share n_AB / (n_AB + n_BA), and the raw count.
- Left-truncation control: a patient's codes recorded on their first eligible era day are excluded from
  precedence counting (Project 1 showed 40% of cardiometabolic carriers are coded on day one). Sensitivity:
  include them.
- Outputs: directed edge list CSV, reciprocity share, and a table of the strongest directed edges into each
  target core and through each bridge code.

**Motif census (classical, feeds the paper).** On Layer D: count the 3-node directed connected subgraphs
(the 13 standard triad classes; the feed-forward chain A -> B -> C is the one of interest). Null: 100
degree-preserving directed rewirings. Report z-scores per motif class overall, and the motif participation of
each bridge code (how often a bridge is the middle node of a chain that ends in a target core). Tooling:
NetworkX triadic census for counts, igraph for rewiring **[default]**. Chains through bridge codes are listed
with their patient counts as a readable table.

**Layer N (null graphs, for experiment D1).** Degree-preserving rewirings of Layer U (reuse
`bridgedx.null_models`), 5 rewired graphs, and one edge-shuffled graph (same edge count, random endpoints).
Node features unchanged. **[rubric]**

**Patient subgraph for a row.** Induced subgraph of the row's masked code set on the chosen layer. Sparse
subgraphs: (a) isolates kept as nodes (they carry features; pooling sees them); (b) ablation adding 1-hop
neighbours from the global graph as context nodes with a flag feature (Chet-style). **[default]** Primary is
(a); (b) is experiment B6.

**Bipartite variant (experiment B2).** Heterogeneous graph per row: one patient node (DEMO + UTIL features)
linked to its code nodes (static + temporal node features), code-code edges from Layer U. Implemented with PyG
`HeteroConv` over SAGE/GAT layers; readout from the patient node. Designed so that, with citations and a
comparison against all baselines, it can be presented as extra-credit Option A ("heterogeneous variant") if
week 3 is on schedule; otherwise it ships as the rubric's bipartite-vs-diagnosis-only ablation. Decision point:
end of week 3. **[decided 2026-11-06]**

## 7. Models

All models see identical rows, splits, and feature definitions. Each is run with 5 seeds; mean and SD.

Non-graph baselines (tuned as carefully as the GNNs; the rubric requires LR or MLP on the same features):
1. Prevalence-only: predicts the development positive rate. Floor.
2. Logistic regression (L2) on CODE (496-node bag) + DEMO + UTIL.
3. Gradient-boosted trees (sklearn HistGradientBoosting **[default]**: no new native dependency) on CODE +
   DEMO + UTIL + BRIDGE row features. Parity baseline.
3b. The same on ALL categories in the window (R, Z, O and sub-floor codes, about 617+ columns) + DEMO + UTIL.
   Strongest realistic tabular baseline; the headline paired comparison is reported against both 3 and 3b.
4. MLP on the same inputs as 3.
4b. MLP on pooled node features: mean and max over the row's codes of (STRUCT + TEMP + BRIDGE node features)
   concatenated with CODE + DEMO + UTIL. Same inputs as the GNN, no message passing. This is the baseline that
   isolates what the graph structure contributes versus what the node features contribute. **[rubric: "same
   input features"]**
5. Bridge-only: logistic regression on BRIDGE row features + DEMO + UTIL. Isolates the bridge claim.

GNNs (PyTorch Geometric):
6. GCN, 7. GraphSAGE (mean aggregator), 8. GAT (4 heads). Default 2 layers, hidden 64, dropout 0.3, readout =
   concatenation of mean and max pooling, concatenated with DEMO + UTIL, then a 2-layer MLP head. Edge weight
   used where the layer supports it (GCN via `edge_weight`; SAGE and GAT unweighted unless the experiment says
   otherwise). Loss: binary cross-entropy with positive-class weight = N_neg / N_pos on the training folds.
   Adam; grid: lr {1e-3, 3e-4}, weight decay {0, 1e-4}, hidden {64}, layers {2} (3 in E1), dropout {0.3},
   batch size {64}; early stopping on fold-validation AUPRC, patience 20, max 300 epochs. Every grid point,
   best configuration, seed list and training curve is logged to `outputs/logs/`. **[rubric]**
9. GNN on Layer D: same three architectures with directed message passing (SAGE and GAT accept a directed
   `edge_index`; GCN uses the symmetrised variant and is reported as such).

**[default]** Separate binary models per target. A multi-task head (one model, four outputs) is optional if
time permits; not primary because eligible row sets differ per target.

**Optional side task (demoted 2026-11-06).** Node classification on the 496-node graph (CCIR chronic flag;
community) with GCN/SAGE/GAT vs LR. Was added on Nov 5 to cover a rubric risk; the Project 2 description
confirms the task is free-form, so this is now a one-day optional item, built only if everything in section 9
is done.

## 8. Evaluation and experiments

**Metrics and reporting** **[rubric]**
- Primary: AUROC and AUPRC on test rows, mean and SD over 5 seeds, plus a patient-level bootstrap (1,000
  resamples of test patients; percentile method, stated) 95% interval for the final configuration. Also F1 at
  the threshold that maximises F1 on out-of-fold predictions, precision, recall, and the class balance per
  target and split.
- Calibration: Brier score, calibration slope and intercept, reliability plot per target.
- Plots and artifacts: training and validation loss curves per model, ROC and PR curves, confusion matrix at
  the chosen threshold, attention weights of GAT for two or three case rows (optional).
- Stratified test results: recorded sex; age under 65 vs 65 and over at index; subgraph-size bins (fewer than
  3 codes, 3 to 9, 10 or more); index year.
- Paired comparison: best GNN vs 3b and vs 4b on the same test rows, bootstrap difference in AUROC and AUPRC
  with interval. This is the sentence the report and paper turn on.
- Interpretation: coefficients or permutation importance of BRIDGE features in the tabular models; the
  classical content (directed edges into cores, motif chains through bridges, precedence) carries the clinical
  reading. GNNExplainer or integrated gradients only if time permits.
- Reporting standards: TRIPOD+AI items relevant to a course report in a half-page appendix; a Kapoor and
  Narayanan style leakage statement in the Graph Construction section.

**Experiments, organised by the rubric's five families.** Unless stated, each runs the best GNN architecture
(chosen on development AUPRC) for the three primary targets, 5 seeds, mean and SD, with the changed factor and
fixed settings in the table caption and a short interpretation. The first row in each family is the minimum.

| ID | Family | Changed factor | Answers |
|---|---|---|---|
| A1 | A features | full vs no BRIDGE (a: features removed; b: bridge codes also removed from input) | "do bridges help predict": the headline |
| A2 | A features | CODE only vs CODE + DEMO + UTIL vs full | demographics and utilisation (rubric) |
| A3 | A features | with vs without STRUCT (P1 structural measures, training graph only) | rubric item verbatim |
| A4 | A features | with vs without TEMP | temporal vs no temporal (rubric) |
| B1 | B structure | Layer U vs Layer D (directed precedence) | Dr. Kim's temporal direction (rubric: co-occurrence vs directed) |
| B2 | B structure | diagnosis-only subgraph vs bipartite patient-diagnosis HeteroGNN | rubric; extra-credit candidate |
| B3 | B structure | weighted vs unweighted edges | rubric |
| B4 | B structure | Bonferroni edges vs BH-FDR edges vs top-10 backbone | threshold and filter sensitivity (rubric) |
| B5 | B structure | P1 full-era graph vs pre-2020 rebuilt graph, rows indexed >= 2020 | time-respecting graph (section 5a) |
| B6 | B structure | lookback 365 vs 1,095 vs all; with vs without 1-hop expansion | observation window; sparse subgraphs |
| C1 | C scale | train on 25% / 50% / 100% of development patients (patient-level subsample, same test) | learning curve, overfitting (rubric) |
| C2 | C subgroup | test performance by sex, age band, subgraph size, index year | rubric |
| D1 | D null model | real Layer U vs 5 degree-preserving rewired graphs vs edge-shuffled graph, node features fixed | does structure carry signal (rubric) |
| D2 | D null model | real node features vs row-wise shuffled node features, edges fixed | does feature content carry signal (optional) |
| E1 | E model | depth 2 vs 3 layers (all three GNNs) | rubric |
| E2 | E model | GraphSAGE mean vs max aggregator; GAT heads 2 vs 4 vs 8 | rubric |
| E3 | E model | hidden 32 vs 64 vs 128 (best GNN) | optional |

Priority order if time runs short: A1, B1, D1, E1, C1, B3, A2, B5, B2, A3, A4, C2 (cheap, always), B4, B6, E2, E3,
D2. The minimum in section 9 is the first row of every family plus A1 and B1.

## 9. Minimum viable deliverable (what must exist on Dec 4 regardless)

1. Label rows, per-row code sets with masking, and feature matrices materialised; node and edge lists and
   thresholds exported as data artifacts (steps 01-02).
2. Baselines 1-4b and GNNs 6-8 on Layer U for the three primary targets, 5 seeds, GroupKFold selection,
   test metrics with intervals, calibration, curves, ROC/PR plots, confusion matrices, logs, hyperparameter
   records.
3. Experiments A1, B1 (one architecture at least), B3, C1, C2, D1, E1.
4. Layer D built and described (directed edges table, precedence through bridges), motif census if time.
5. Report (6-8 pages, seven sections) with leakage statement, limitations; README; reproducible run.

Everything else in this document is ordered after that.

## 10. Repository and pipeline shape

**[default]** New repository `project2-bridge-gnn` beside `project1-bridge-dx`; Project 1 stays frozen as
the course submission snapshot and is a path dependency (category map, null models, graph builders for B5).
Same conventions: `uv`, `config/p2.yaml` holding every parameter with its reason, numbered scripts writing to
`outputs/`, library in `src/bridgegnn/`, `tests/` on synthetic data, `.gitignore` excluding all patient-level
files. New dependencies: `torch`, `torch_geometric`, `scikit-learn` (present).

| Step | Script | Writes | Owner |
|---|---|---|---|
| 00 | `00_check_inputs.py`: verify P1 outputs, split file, DuckDB cache; print label counts and class balance | nothing | N |
| 01 | `01_rows.py`: join `p2_labels.parquet` (core, 12 mo, uncensored) with per-row code sets (all categories, per lookback, with first/last dates) from `condition_occurrence` via DuckDB; patient covariates; target masking | `rows.parquet`, `row_codes.parquet` | N |
| 02 | `02_features.py`: node feature matrix (static + BRIDGE join from bridges.csv), standardisation fit on development, bag-of-codes matrices (496 and all), pooled-feature matrices, feature-group manifest | `node_x.npy`, `tabular_*.parquet`, `feature_groups.json`, exported node/edge lists and thresholds | D |
| 03 | `03_directed_layer.py`: precedence edges, binomial test, left-truncation control, directed edge CSV, reciprocity, edges into cores; U-bh variant; null graphs (Layer N) | `edges_directed.csv`, `edges_bh.csv`, `null_graphs/`, tables | N |
| 04 | `04_motifs.py`: triad census on Layer D vs 100 rewired nulls; bridge-centred chains | `motifs.csv`, `bridge_chains.csv`, figure | N |
| 05 | `05_baselines.py`: models 1-5 (incl. 3b, 4b) per target, 5 seeds, GroupKFold selection, refit, test once | `results_baselines.csv`, predictions, logs | D |
| 06 | `06_gnn.py --layer U|D|N --arch gcn|sage|gat --features <groups> --depth --agg --heads --lookback --expand --weighted`: PyG dataset builder (one `Data` per row, cached), GroupKFold selection, refit, test once, curves and logs | `results_gnn.csv`, predictions, curves, checkpoints (ignored) | N + D |
| 06b | `06b_hetero.py`: bipartite HeteroGNN (B2) | `results_hetero.csv` | N (week 3) |
| 07 | `07_experiments.py`: drives 05/06 over the experiment grid in section 8, incl. C1 subsampling and D1 null graphs | `results_experiments.csv` | D |
| 08 | `08_evaluate.py`: bootstrap intervals, calibration, F1 threshold, confusion matrices, stratified tables, paired comparisons, all plots, LaTeX tables | `tables/*.tex`, `fig_*.pdf` | N |
| 09 | `09_pre2020_graph.py`: rebuild P1 graph and node features from development records before 2020-01-01 via the P1 code; then B5 and the temporal backtest | `outputs/pre2020/`, `results_b5.csv`, `results_backtest.csv` | N |
| 10 | report repo `project2-bridge-gnn-report` (IEEE, 6-8 pages, same Makefile pattern, number macros) | PDF | both |

Owners are a proposal for the Dylan conversation **[open]**; the course requires visible contribution from
both. Dylan has a review clone of the P1 repo and knows the config and outputs layout.

Tests: synthetic 20-node graph and 200 synthetic rows; leakage guards (section 5); label join counts; target
masking; directed-edge test behaviour on a planted A -> B signal; each GNN trains one epoch and produces finite
predictions; evaluation code reproduces known AUROC and bootstrap interval on a toy case; null-graph builder
preserves degrees.

## 11. Timeline (four weeks)

| Week | Dates | Done by end |
|---|---|---|
| 1 | Nov 6-12 | Design approved; repo scaffold; steps 00-02; baselines 1-4b on one target; Layer D first version; related-works list started (5+) |
| 2 | Nov 13-19 | GCN/SAGE/GAT on Layer U for all primary targets; experiment driver; A1, B1, B3, D1, E1 running; evaluation script with bootstrap, calibration, plots |
| 3 | Nov 20-26 | C1, C2, A2, B5 (pre-2020 rebuild), motif census; B2 HeteroGNN and the extra-credit decision; report skeleton with numbers as macros; Thanksgiving Nov 26 |
| 4 | Nov 27-Dec 3 | Remaining experiments if time; writing to 6-8 pages; reproducibility run from scratch; submit Dec 3 |

## 12. What this hands to the paper (spring)

- A clean prediction result, positive or negative, with the bridge ablation as the headline and the null-model
  and pooled-feature baselines as the evidence that the comparison was fair.
- The directed precedence layer and motif census as new classical analysis beyond Project 1.
- A reproducible pipeline that a second dataset (TGH / TriNetX) could be dropped into.
- Open items carried forward, not solved here: burden-matched null for Project 1 bridges; adaptive permutation
  for subgroup edge tests; weight sensitivity (Jaccard, lift); rolling time-respecting graphs; larger cohorts.

## 13. Risks and how the design absorbs them

| Risk | Response |
|---|---|
| GNN does not beat tabular baselines | Expected at this scale. Report honestly; the paired bootstrap difference is the result; D1 and 4b show whether structure or features carried what signal there is. The directed/motif analysis still carries clinical content. |
| cancer_metastatic too few positives | Secondary target, flagged, no tuning on it. |
| Patient subgraphs too sparse for message passing | 1,095-day primary lookback; isolates kept; B6 expansion; subgraph-size strata make the effect visible. |
| Directed layer inherits left truncation | Day-one codes excluded from precedence counts; sensitivity reported. |
| Motif census on a dense graph is uninformative | Run on Layer D (already thresholded), not on the undirected graph; if saturated, use the backbone. |
| Multiple rows per patient inflate apparent sample size | Patient-level split and bootstrap; row and patient counts both reported. |
| Time-level graph contamination questioned by the grader | Section 5a wording in the leakage statement; B5 shows the result on a pre-2020 graph. |
| Experiment grid too large for four weeks | Section 8 priority order; section 9 minimum; experiment driver makes each run a config line. |
| PyG on Apple Silicon | CPU is sufficient; MPS optional; pin versions in `uv.lock`. |
| Two-person bandwidth | Dylan owns steps 02, 05, 07 (self-contained); minimum fixed. |

## 14. Decisions left to the builder

Exact grids beyond those stated; PyG dataset caching format; figure styling; LaTeX table layout; igraph vs
NetworkX for directed rewiring; how to batch subgraphs (standard PyG `DataLoader`); seed values (document in
config); log format.

## 15. Open questions for the humans

1. Dylan's blocks and whether the owner column in section 10 is acceptable to him. **[open]**
2. Should cancer_metastatic be dropped from the report entirely rather than shown as underpowered? **[open,
   default: shown as secondary]**
3. Does Dr. Kim expect the directed layer to replace the undirected graph as the primary model input, or to be
   an experiment as designed here (B1)? **[open, default: B1; ask at office hours]**
4. Extra credit: promote B2 to Option A or not. **[decision point end of week 3]**

## 16. Review and rubric resolutions

| Question | Resolution | Who decided | When |
|---|---|---|---|
| Graph-classification vs the Lecture 05 node-classification recipe | Main task kept; P2 description confirms free choice; side task demoted to optional | Nithin | Nov 5-6 |
| Train/val rows scored on a graph that saw their own future records | Accept, document; pre-2020 rebuild promoted to experiment B5 | Nithin | Nov 5-6 |
| Fixed train/val vs cross-validation in train+val | 5-fold GroupKFold in development set, refit, test once | default | Nov 5 |
| What counts as a bridge in A1 and baseline 5 | B-flag + B-score pre-registered; power limit stated | default | Nov 5 |
| Tabular baselines on 496 nodes only vs all codes | Step 01 extracts all codes; baseline 3b added | default | Nov 5 |
| Rubric: non-graph baseline on the same input features | Baseline 4b (MLP on pooled node features) added | default | Nov 6 |
| Rubric: five experiment families | Section 8 reorganised; C1, D1, E1, E2, B4 added; minimum = first row per family | default | Nov 6 |
| Rubric: temporal features ablation | TEMP feature group added (days since last/first recording, recent flag); A4 | default | Nov 6 |
| Rubric: exclude target and near-synonym codes from inputs | Inputs mask core + chapter block; eligibility unchanged | default | Nov 6 |
| Rubric: bipartite vs diagnosis-only | B2 as HeteroGNN, doubling as extra-credit Option A candidate | Nithin | Nov 6 |
| Rubric: labs feature ablation | Skipped with reason (labs not used) | default | Nov 6 |
| Rubric: report length and sections | 6-8 pages, seven sections, 5+ related works | rubric | Nov 6 |
