import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from bridgegnn import compare

REPO = Path(__file__).resolve().parents[1]


def write(root: Path, rel: str, df: pl.DataFrame) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(root / rel)


def gnn_rows(auroc=0.61, params='{"lr": 0.001}'):
    return pl.DataFrame({"run_id": ["G-sage", "G-sage"], "target": ["cardiometabolic", "mental_health"], "seed": [0, 0],
                         "model": ["sage", "sage"], "params_json": [params, params], "test_auroc": [auroc, 0.55]})


def baseline_rows(auroc=0.61):
    return pl.DataFrame({"run_id": ["B-lr"], "target": ["cardiometabolic"], "seed": [0], "model": ["lr"],
                         "params_json": ['{"C": 1}'], "test_auroc": [auroc]})


def two_trees(tmp_path, new_gnn=None, new_base=None):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old, "results/gnn.csv", gnn_rows()); write(new, "results/gnn.csv", new_gnn if new_gnn is not None else gnn_rows())
    write(old, "results/baselines.csv", baseline_rows()); write(new, "results/baselines.csv", new_base if new_base is not None else baseline_rows())
    return old, new


def status_of(df, file, column):
    r = df.filter((pl.col("file") == file) & (pl.col("column") == column))
    assert r.height == 1, (file, column, df)
    return r["status"][0]


def test_compare_identical_trees_are_exact(tmp_path):
    old, new = two_trees(tmp_path)
    df = compare.compare_dirs(old, new)
    assert set(df["status"].to_list()) == {"exact"}
    assert compare.summary(df)["exact"] == df.height


def test_compare_within_tolerance_for_torch_rows_only(tmp_path):
    old, new = two_trees(tmp_path, new_gnn=gnn_rows(auroc=0.62), new_base=baseline_rows(auroc=0.62))
    df = compare.compare_dirs(old, new, tol=0.02)
    assert status_of(df, "results/gnn.csv", "test_auroc") == "within_tol"
    assert status_of(df, "results/baselines.csv", "test_auroc") == "exceeds"
    assert compare.compare_dirs(old, new, tol=0.005).filter(pl.col("file") == "results/gnn.csv")["status"].to_list().count("exceeds") == 1


def test_compare_params_json_must_match_exactly(tmp_path):
    old, new = two_trees(tmp_path, new_gnn=gnn_rows(params='{"lr": 0.003}'))
    df = compare.compare_dirs(old, new)
    assert status_of(df, "results/gnn.csv", "params_json") == "exceeds"
    assert status_of(df, "results/gnn.csv", "test_auroc") == "exact"


def test_compare_reports_missing_file_and_column(tmp_path):
    old, new = two_trees(tmp_path)
    write(old, "motifs/triad_census.csv", pl.DataFrame({"triad": ["030T"], "z": [1.0]}))
    write(old, "eval/tables/runs_agg.csv", pl.DataFrame({"run_id": ["G-sage"], "target": ["mental_health"], "auroc_mean": [0.5], "brier_mean": [0.05]}))
    write(new, "eval/tables/runs_agg.csv", pl.DataFrame({"run_id": ["G-sage"], "target": ["mental_health"], "auroc_mean": [0.5]}))
    df = compare.compare_dirs(old, new)
    assert status_of(df, "motifs/triad_census.csv", "*") == "missing_file"
    assert status_of(df, "eval/tables/runs_agg.csv", "brier_mean") == "missing_column"
    assert status_of(df, "eval/tables/runs_agg.csv", "auroc_mean") == "exact"


def test_compare_rows_that_do_not_align_exceed(tmp_path):
    extra = pl.concat([gnn_rows(), gnn_rows().with_columns(pl.lit(1, dtype=pl.Int64).alias("seed"))])
    old, new = two_trees(tmp_path, new_gnn=extra)
    df = compare.compare_dirs(old, new)
    r = df.filter((pl.col("file") == "results/gnn.csv") & (pl.col("column") == "*"))
    assert r["status"][0] == "exceeds" and "rows differ" in r["note"][0]


def test_compare_npy_matrix(tmp_path):
    old, new = two_trees(tmp_path)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(old / "node_x.npy", x); np.save(new / "node_x.npy", x)
    assert status_of(compare.compare_dirs(old, new), "node_x.npy", "matrix") == "exact"
    np.save(new / "node_x.npy", x + 1e-3)
    assert status_of(compare.compare_dirs(old, new), "node_x.npy", "matrix") == "exceeds"


def test_cli_exit_code_and_markdown(tmp_path):
    old, new = two_trees(tmp_path)
    cmd = [sys.executable, str(REPO / "scripts/compare_runs.py"), str(old), str(new), "--md", str(tmp_path / "c.md")]
    assert subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO).returncode == 0
    assert "| file |" in (tmp_path / "c.md").read_text()
    old, new = two_trees(tmp_path / "b", new_base=baseline_rows(auroc=0.7))
    cmd = [sys.executable, str(REPO / "scripts/compare_runs.py"), str(old), str(new)]
    assert subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO).returncode == 1


def test_compare_equal_infinite_and_null_cells_are_exact(tmp_path):
    old, new = two_trees(tmp_path)
    df = pl.DataFrame({"triad": ["030T", "021D"], "z": [float("inf"), None], "p": [0.0, 1.0]})
    write(old, "motifs/triad_census.csv", df); write(new, "motifs/triad_census.csv", df)
    out = compare.compare_dirs(old, new)
    assert status_of(out, "motifs/triad_census.csv", "z") == "exact"


def test_compare_null_key_cells_still_align(tmp_path):
    old, new = two_trees(tmp_path)
    df = pl.DataFrame({"run": ["main", "g0", None, None], "drop_day_one": [True, True, False, None], "n_edges": [1, 2, 3, 4]})
    write(old, "layers/directed_summary.csv", df); write(new, "layers/directed_summary.csv", df)
    out = compare.compare_dirs(old, new).filter(pl.col("file") == "layers/directed_summary.csv")
    assert set(out["status"].to_list()) == {"exact"}


def test_compare_params_json_ignores_path_and_identity_keys(tmp_path):
    a = '{"lr": 0.001, "node_features_dir": "/Users/a/project2-bridge-gnn/outputs/features", "layer_file": "/Users/a/x/outputs/pre2020/graph.graphml", "run_id": "G-sage"}'
    b = '{"lr": 0.001, "node_features_dir": "/Volumes/b/project2-bridge-gnn-rerun/outputs/features_xcheck", "layer_file": "/tmp/y/outputs/pre2020/graph.graphml", "run_id": "G-sage"}'
    old, new = two_trees(tmp_path, new_gnn=gnn_rows(params=b))
    write(old, "results/gnn.csv", gnn_rows(params=a))
    assert status_of(compare.compare_dirs(old, new), "results/gnn.csv", "params_json") == "exact"
    c = b.replace('"lr": 0.001', '"lr": 0.0003')
    write(new, "results/gnn.csv", gnn_rows(params=c))
    assert status_of(compare.compare_dirs(old, new), "results/gnn.csv", "params_json") == "exceeds"


def test_compare_npy_only_main_matrix(tmp_path):
    assert compare.NPY_FILES == ["node_x.npy"]


def test_compare_params_json_ignores_null_valued_keys(tmp_path):
    """Flags added to the CLI later are recorded as null; a null flag is the same configuration as an absent one."""
    a = '{"lr": 0.001, "weighted": 1}'
    b = '{"lr": 0.001, "weighted": 1, "index_from": null, "test_index_from": null}'
    old, new = two_trees(tmp_path, new_gnn=gnn_rows(params=b))
    write(old, "results/gnn.csv", gnn_rows(params=a))
    assert status_of(compare.compare_dirs(old, new), "results/gnn.csv", "params_json") == "exact"
