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
