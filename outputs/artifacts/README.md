# Aggregate results (no patient-level data)

Category-level and run-level outputs of the Project 2 pipeline, copied from `outputs/` for the submission.
Patient-level files (`rows.parquet`, `row_codes.parquet`, `predictions/*.parquet`, subsample lists) are never committed.

| Folder | Contents | Produced by |
|---|---|---|
| `nodes.csv`, `graph_edges.csv`, `node_index.csv`, `node_x.npy`, `feature_groups.json`, `construction_thresholds.json` | Main graph with aligned diagnosis-level feature columns and its construction rules | step 02 |
| `results/gnn.csv` | one row per (run_id, target, seed): test AUROC / AUPRC / Brier, dev OOF AUPRC, flags and hyperparameters in `params_json` | step 06 |
| `eval/tables/` | per-run metrics with patient-bootstrap intervals, aggregated runs, reliability bins, strata, paired deltas (CSV + LaTeX) | step 08 |
| `eval/figures/` | ROC/PR, reliability and training-curve figures | step 08 |
| `layers/` | directed precedence edges (+ sensitivities), BH edges, layer statistics, directed edges into cores and through bridges | step 03 |
| `motifs/` | triad census vs two nulls, bridge middle shares, bridge chains, motif figure | step 04 |
| `pre2020/` | time-respecting rebuild summary, target definitions with anchor co-assignment, bridges, node features | step 09 |

Experiment ids follow `config/experiments.yaml`; G-* = main comparison, B5-*/BT-* = time-respecting graph and backtest.
