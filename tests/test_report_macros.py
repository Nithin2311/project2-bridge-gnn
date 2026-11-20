import polars as pl
import pytest

from bridgegnn import report_macros

MAIN = ["G-gcn", "G-sage", "G-gat", "B-prevalence", "B-lr", "B-hgb", "B-hgb_all", "B-mlp", "B-mlp_pool", "B-bridge_lr"]
PRIMARY = ["cardiometabolic", "cancer_hematologic", "mental_health"]


def write(root, rel, df):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(root / rel)


def tree(tmp_path, n_seeds_override=None):
    a = tmp_path / "artifacts"
    rows = []
    for run in MAIN + ["A1a"]:
        for t in PRIMARY:
            rows.append({"run_id": run, "target": t, "model": "sage", "auroc_mean": 0.6123, "auroc_sd": 0.0210, "auprc_mean": 0.1, "auprc_sd": 0.01,
                         "brier_mean": 0.05, "n_seeds": 5, "n": 1752, "pos": 91, "auroc_seed": 0.61, "auroc_lo": 0.55, "auroc_hi": 0.67,
                         "auprc_seed": 0.1, "auprc_lo": 0.07, "auprc_hi": 0.15})
    agg = pl.DataFrame(rows)
    if n_seeds_override:
        run, t, n = n_seeds_override
        agg = agg.with_columns(pl.when((pl.col("run_id") == run) & (pl.col("target") == t)).then(n).otherwise(pl.col("n_seeds")).alias("n_seeds"))
    write(a, "eval/tables/runs_agg.csv", agg)
    write(a, "eval/tables/paired.csv", pl.DataFrame({"run_id": ["G-sage"], "baseline": ["B-hgb_all"], "target": ["cardiometabolic"], "seed": [0],
                                                      "n": [1752], "d_auroc": [0.0801], "d_auroc_lo": [-0.002], "d_auroc_hi": [0.16],
                                                      "d_auprc": [0.01], "d_auprc_lo": [-0.03], "d_auprc_hi": [0.06]}))
    write(a, "layers/layer_stats.csv", pl.DataFrame({"layer": ["U", "D", "U-bh", "U-bb"], "n_nodes": [496] * 4, "n_edges": [9115, 1404, 28862, 2000],
                                                     "n_isolates": [10, 213, 0, 0], "bh_family_n": [None, None, 190036, None]}))
    write(a, "motifs/triad_census.csv", pl.DataFrame({"triad": ["030T", "021D"], "observed": [100, 50], "z": [5.5, 1.0], "p_emp": [0.0, 0.3],
                                                      "z_u": [0.97, 0.5], "p_emp_u": [0.40, 0.6]}))
    write(a, "pre2020/summary.csv", pl.DataFrame({"n_edges_new": [5263], "n_edges_u": [9115], "edge_jaccard": [0.5427], "n_dev_patients": [2304],
                                                  "empty_cores": ["cardiometabolic"]}))
    write(a, "results/gnn.csv", pl.DataFrame({"run_id": ["G-sage", "A1a"], "target": ["cardiometabolic"] * 2, "seed": [0, 0], "test_auroc": [0.6, 0.6]}))
    return a


def test_macros_emit_res_and_pair_entries(tmp_path):
    out = report_macros.build(tree(tmp_path), None)
    assert r"\expandafter\def\csname res:G-sage:cardiometabolic:auroc_mean\endcsname{0.612}" in out
    assert r"\csname res:G-sage:cardiometabolic:n\endcsname{1752}" in out
    assert r"\csname pair:G-sage:B-hgb_all:cardiometabolic:d_auroc\endcsname{+0.080}" in out
    assert r"\newcommand{\res}[3]" in out and r"\newcommand{\pair}[4]" in out


def test_macros_require_five_seeds_for_main_runs(tmp_path):
    with pytest.raises(ValueError, match="G-sage"):
        report_macros.build(tree(tmp_path, n_seeds_override=("G-sage", "cardiometabolic", 4)), None)
    report_macros.build(tree(tmp_path / "ok", n_seeds_override=("A1a", "cardiometabolic", 4)), None)   # non-main runs may be partial


def test_macros_escape_latex_specials():
    assert report_macros.tex_escape("a_b 25% #1 & c") == r"a\_b 25\% \#1 \& c"


def test_macros_scalars_from_tables(tmp_path):
    out = report_macros.build(tree(tmp_path), {"nRows": 20517, "nDevPatients": 2017})
    for line in [r"\newcommand{\nEdgesU}{9115}", r"\newcommand{\nEdgesD}{1404}", r"\newcommand{\nEdgesBH}{28862}", r"\newcommand{\nIsolatesD}{213}",
                 r"\newcommand{\motifZ}{5.50}", r"\newcommand{\motifZu}{0.97}", r"\newcommand{\motifPu}{0.40}",
                 r"\newcommand{\nEdgesPre}{5263}", r"\newcommand{\preJaccard}{0.54}", r"\newcommand{\nDevPre}{2304}",
                 r"\newcommand{\nGnnRuns}{2}", r"\newcommand{\nRows}{20517}", r"\newcommand{\nDevPatients}{2017}"]:
        assert line in out, line


def test_cli_writes_file(tmp_path):
    a = tree(tmp_path)
    out = tmp_path / "macros.tex"
    report_macros.main(["--artifacts", str(a), "--out", str(out)])
    assert out.exists() and "res:G-gat:mental_health:auroc_mean" in out.read_text()


def test_macros_emit_strata_entries(tmp_path):
    a = tree(tmp_path)
    write(a, "eval/tables/strata.csv", pl.DataFrame({"stratum": ["sex", "sex", "sex"], "level": ["female", "male", "female"],
                                                     "n": [1039, 713, 1039], "pos": [50, 41, 50], "auroc": [0.5722, 0.7546, 0.6],
                                                     "auprc": [0.07, 0.17, 0.08], "run_id": ["G-sage"] * 3, "target": ["cancer_hematologic"] * 3,
                                                     "seed": [0, 0, 1]}))
    out = report_macros.build(a, None)
    assert r"\csname strat:G-sage:cancer_hematologic:sex:female:auroc\endcsname{0.572}" in out   # first seed only
    assert r"\csname strat:G-sage:cancer_hematologic:sex:male:pos\endcsname{41}" in out
    assert r"\newcommand{\strat}[5]" in out


def test_macros_pair_uses_the_pooled_row_and_counts_seeds_excluding_zero(tmp_path):
    a = tree(tmp_path)
    rows = [{"run_id": "G-sage", "baseline": "B-hgb_all", "target": "mental_health", "seed": s, "n": 1375,
             "d_auroc": d, "d_auroc_lo": lo, "d_auroc_hi": hi, "d_auprc": 0.0, "d_auprc_lo": -0.03, "d_auprc_hi": 0.03}
            for s, d, lo, hi in [(-1, -0.0612, -0.121, -0.0015), (0, -0.068, -0.141, 0.002), (1, -0.11, -0.18, -0.04),
                                 (2, -0.12, -0.19, -0.05), (3, -0.10, -0.17, -0.03), (4, -0.09, -0.16, -0.02)]]
    write(a, "eval/tables/paired.csv", pl.DataFrame(rows))
    out = report_macros.build(a, None)
    assert r"\csname pair:G-sage:B-hgb_all:mental_health:d_auroc\endcsname{-0.061}" in out      # pooled row, not seed 0
    assert r"\csname pair:G-sage:B-hgb_all:mental_health:n_excl\endcsname{4}" in out            # per-seed intervals excluding zero
    assert r"\csname pair:G-sage:B-hgb_all:mental_health:n_seeds\endcsname{5}" in out
    assert r"\csname pair:G-sage:B-hgb_all:mental_health:seed\endcsname{pooled}" in out


def test_macros_res_includes_interval_width(tmp_path):
    out = report_macros.build(tree(tmp_path), None)
    assert r"\csname res:G-sage:cardiometabolic:auroc_ciw\endcsname{0.120}" in out   # hi 0.67 - lo 0.55


def test_macros_sex_gap_and_dev_oof_lookups(tmp_path):
    a = tree(tmp_path)
    write(a, "eval/tables/sex_gap.csv", pl.DataFrame({"run_id": ["G-sage"], "target": ["cancer_hematologic"], "n_female": [1039], "n_male": [713],
                                                      "pos_female": [50], "pos_male": [41], "auroc_female": [0.5722], "auroc_male": [0.7546],
                                                      "gap": [-0.1824], "gap_lo": [-0.31], "gap_hi": [-0.05], "n_boot": [1000]}))
    write(a, "results/gnn.csv", pl.DataFrame({"run_id": ["G-sage"] * 2, "target": ["cardiometabolic"] * 2, "seed": [0, 1],
                                              "dev_oof_auprc": [0.26, 0.28], "test_auprc": [0.16, 0.16], "test_auroc": [0.6, 0.6]}))
    write(a, "results/baselines.csv", pl.DataFrame({"run_id": ["B-lr"], "target": ["cardiometabolic"], "seed": [0],
                                                    "dev_oof_auprc": [0.17], "test_auprc": [0.197], "test_auroc": [0.6]}))
    out = report_macros.build(a, None)
    assert r"\csname sexgap:G-sage:cancer_hematologic:gap\endcsname{-0.182}" in out
    assert r"\csname sexgap:G-sage:cancer_hematologic:pos_female\endcsname{50}" in out
    assert r"\newcommand{\sexgap}[3]" in out
    assert r"\csname res:G-sage:cardiometabolic:dev_oof_auprc\endcsname{0.270}" in out
    assert r"\csname res:B-lr:cardiometabolic:dev_oof_auprc\endcsname{0.170}" in out


def test_macros_confusion_counts_from_first_seed(tmp_path):
    a = tree(tmp_path)
    write(a, "eval/tables/runs.csv", pl.DataFrame({"run_id": ["G-sage", "G-sage"], "target": ["cancer_hematologic"] * 2, "seed": [1, 0],
                                                   "tp": [40, 42], "fp": [700, 650], "fn": [51, 49], "tn": [961, 1011], "threshold": [0.1, 0.09],
                                                   "precision": [0.05, 0.06], "recall": [0.44, 0.46], "f1": [0.09, 0.1]}))
    out = report_macros.build(a, None)
    assert r"\csname res:G-sage:cancer_hematologic:tp\endcsname{42}" in out and r"\csname res:G-sage:cancer_hematologic:tn\endcsname{1011}" in out
    assert r"\csname res:G-sage:cancer_hematologic:recall0\endcsname{0.460}" in out


def test_macros_emit_bh_q_values_over_the_pooled_pairings(tmp_path):
    a = tree(tmp_path)
    rows = []
    for i, (run, d, lo, hi) in enumerate([("G-sage", -0.118, -0.192, -0.047), ("A1a", -0.036, -0.070, -0.004), ("B1", 0.024, -0.016, 0.063)]):
        rows.append({"run_id": run, "baseline": "B-cov_lr", "target": "mental_health", "seed": -1, "n": 1375,
                     "d_auroc": d, "d_auroc_lo": lo, "d_auroc_hi": hi, "d_auprc": 0.0, "d_auprc_lo": -0.03, "d_auprc_hi": 0.03})
        rows.append({**rows[-1], "seed": 0})
    write(a, "eval/tables/paired.csv", pl.DataFrame(rows))
    out = report_macros.build(a, None)
    assert r"\csname pair:G-sage:B-cov_lr:mental_health:q_bh\endcsname{" in out
    import re
    q = {m.group(1): float(m.group(2)) for m in re.finditer(r"pair:(\S+?):B-cov_lr:mental_health:q_bh\\endcsname\{([0-9.]+)\}", out)}
    assert q["G-sage"] < 0.01 and 0.03 < q["A1a"] < 0.06 and q["B1"] > q["A1a"]   # family of three here; ~100 in the real table
    assert r"\newcommand{\nPairings}{3}" in out


def test_macros_censored_counts_from_label_counts(tmp_path):
    a = tree(tmp_path)
    write(a, "cohort/label_counts.csv", pl.DataFrame({"target": ["cardiometabolic", "cardiometabolic"], "split": ["all", "test"],
                                                     "rows": [1863, 664], "positives": [163, 56], "censored": [182, 70]}))
    out = report_macros.build(a, None)
    assert r"\newcommand{\nCensoredCM}{182}" in out and r"\newcommand{\nCensoredTestCM}{70}" in out
