# Data contract

Every file that crosses between pipeline steps or between people is listed here. The authoritative version is
`src/bridgegnn/contract.py` (`SCHEMAS`); this page is the prose copy. **Rule: write with `contract.write(df,
path, name)` and read with `contract.read(path, name)`. Both validate.** A schema change is a PR that touches
`contract.py`, this page, and every producer and consumer.

Conventions: `person_id` is Int64 (OMOP person id). Dates are polars `Date`. Categories are 3-character ICD-10
strings (`I10`). `split` is one of `train`, `val`, `test` from Project 1's `outputs/patient_split.csv`
(canonical EHRShot split). `target` is one of `cardiometabolic`, `cancer_metastatic`, `cancer_hematologic`,
`mental_health`.

## Files and who writes them

| File | Written by | Read by | Contract name |
|---|---|---|---|
| `outputs/rows.parquet` | step 01 (Nithin) | 02, 05, 06, 08 | `rows` |
| `outputs/row_codes.parquet` | step 01 | 02, 06 | `row_codes` |
| `outputs/features/node_index.csv` | step 02 (Dylan) | 06, 08 | `node_index` |
| `outputs/features/node_x.npy` | step 02 | 06 | float32 `[n_nodes, n_features]`, rows in `node_index` order |
| `outputs/features/feature_groups.json` | step 02 | 05, 06, 08 | `{group: [column names]}` for `STRUCT`, `BRIDGE`, `CHAPTER`, `COMMUNITY` (node columns) and `CODE`, `ALL`, `DEMO`, `UTIL`, `TEMP`, `POOL` (row columns) |
| `outputs/features/tabular_{target}_{split}.parquet` | step 02 | 05 | `tabular` |
| `outputs/artifacts/*.csv`, `construction_thresholds.json` | step 02 | report | aggregate deliverables (rubric) |
| `outputs/predictions/{run_id}_{target}_seed{seed}.parquet` | 05, 06 | 08 | `predictions` |
| `outputs/results/*.csv` | 05, 06, 07 | 08 | `results` |
| `outputs/logs/*.log` | 05, 06, 07 | humans | free text |

## Schemas

### `rows` — one row per (patient, index date, target); only task rows (core definition, 12 months, uncensored)

| column | dtype | meaning |
|---|---|---|
| person_id | Int64 | |
| index_date | Date | prediction time point (landmark grid, 182-day step from 2017-10-01) |
| target | Utf8 | |
| split | Utf8 | train / val / test |
| positive | Boolean | first target-core code within (index, index + 12 mo] |
| days_to_event | Int64 | null when negative |
| age_at_index | Float64 | (index_date - birth_date) / 365.25 |
| sex | Utf8 | female / male |
| index_year | Int32 | |
| days_since_first_dx | Int32 | index_date minus the patient's first ICD-10-era diagnosis date |
| n_codes_lb365 | Int32 | distinct main-graph categories with a recording in (index - 365 d, index]; before masking |
| n_codes_lb1095 | Int32 | same, 1,095 days |
| n_codes_lball | Int32 | same, all era history |
| n_rows_lb1095 | Int32 | diagnosis rows (not distinct) in the 1,095-day window; utilisation proxy |
| n_visit_days_365 | Int32 | distinct visit dates in (index - 365 d, index] |

### `row_codes` — one row per (patient, index date, category); shared across targets, masking per target

| column | dtype | meaning |
|---|---|---|
| person_id, index_date | Int64, Date | key with category |
| category | Utf8 | any ICD-10 category recorded on or before the index (era only) |
| first_date, last_date | Date | first and most recent recording on or before index |
| days_since_first, days_since_last | Int32 | index minus those dates (>= 0) |
| in_lb365, in_lb1095 | Boolean | last_date inside the window; all-history membership is implied by the row existing |
| in_main_graph | Boolean | category is one of the 496 Project 1 nodes |
| masked_{target} | Boolean x 4 | category is in that target's stable core or chapter block (excluded from inputs for that target) |

### `node_index` — `category` Utf8, `idx` Int32 (0..n-1); row `idx` of `node_x.npy` is that category.

### `tabular` — base columns `person_id, index_date, target, split, positive` plus feature columns

Feature columns must start with one of: `c_` (CODE bag over the 496 main-graph nodes, masked for the table's
target), `all_` (bag over every category with >= 10 development rows, masked), `demo_`, `util_`, `temp_`,
`bridge_`, `pool_mean_`, `pool_max_`. The schema of the three split files for one target is identical.

### `predictions`

| column | dtype |
|---|---|
| run_id, model, target | Utf8 |
| seed | Int32 |
| person_id | Int64 |
| index_date | Date |
| split | Utf8 (dev rows carry out-of-fold predictions; test rows carry the refit model's) |
| y | Boolean |
| p | Float64 |

### `results` — one row per (run, target, seed)

| column | dtype |
|---|---|
| run_id, family, model, target | Utf8 (family: baseline, gnn, A, B, C, D, E) |
| seed | Int32 |
| layer | Utf8 (none, U, U-bb, U-bh, D, N) |
| features | Utf8 (comma-joined feature groups used) |
| params_json | Utf8 (best configuration) |
| dev_oof_auprc, test_auroc, test_auprc, test_brier | Float64 |
| n_test_rows, n_test_pos | Int32 |

## Feature groups

| group | where | columns |
|---|---|---|
| CODE | row | `c_<cat>` for the 496 main-graph nodes |
| ALL | row | `all_<cat>` for every category with >= 10 development rows |
| DEMO | row | `demo_age`, `demo_female` |
| UTIL | row | `util_n_codes`, `util_n_rows`, `util_n_visit_days`, `util_days_since_first_dx`, `util_index_year` |
| TEMP | row (aggregate) and node (per code, step 06) | `temp_min_days_since_last`, `temp_mean_days_since_last`, `temp_share_recent`; per-node: days since last / first recording, recent flag |
| STRUCT | node | Project 1 structural measures (degree, strength, betweenness, closeness, eigenvector, PageRank, participation, within-module z, degree residual, inpatient share, CCIR chronic share, mean age, female share, backbone degree) plus chapter and community one-hots |
| BRIDGE | node and row | node: `bridge_flag`, `bridge_flag_<target>`, `bridge_score_<target>`; row: `bridge_n_flag`, `bridge_sum_score`, `bridge_max_score` |
| POOL | row | `pool_mean_<f>`, `pool_max_<f>` over the row's unmasked main-graph codes for every STRUCT and BRIDGE node column |

## Leakage rules the contract enforces or supports

- Test rows never enter a fit: functions that fit take a `dev_mask` and assert it.
- Masked codes are absent from every input column for that target.
- Node features and edges are Project 1 train+val artifacts; test patients never contributed to them.
