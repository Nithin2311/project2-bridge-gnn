from pathlib import Path

import pytest

from bridgegnn import artifacts


def test_collect_copies_aggregates_and_refuses_patient_files(tmp_path):
    out = tmp_path / "outputs"
    for rel in ["results/gnn.csv", "eval/tables/runs_agg.csv", "eval/figures/roc.pdf", "layers/edges_bh.csv",
                "motifs/triad_census.csv", "pre2020/summary.csv", "features/node_index.csv", "features/node_x.npy",
                "layers/tables/layer_stats.csv"]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b"x")
    (out / "rows.parquet").write_bytes(b"x")
    (out / "predictions").mkdir()
    (out / "predictions/a.parquet").write_bytes(b"x")
    copied = artifacts.collect(out, tmp_path / "dest")
    rel = sorted(str(p.relative_to(tmp_path / "dest")) for p in copied)
    assert rel == ["eval/figures/roc.pdf", "eval/tables/runs_agg.csv", "layers/edges_bh.csv", "layers/layer_stats.csv",
                   "motifs/triad_census.csv", "node_index.csv", "node_x.npy", "pre2020/summary.csv", "results/gnn.csv"]
    assert not list((tmp_path / "dest").rglob("*.parquet"))


def test_collect_refuses_parquet_in_the_set():
    with pytest.raises(ValueError):
        artifacts._check_safe(Path("results/x.parquet"))
    with pytest.raises(ValueError):
        artifacts._check_safe(Path("subsamples/C1-25_seed0.txt"))


def test_collect_refuses_two_sources_for_one_destination(tmp_path):
    out = tmp_path / "outputs"
    for rel in ["layers/layer_stats.csv", "layers/tables/layer_stats.csv"]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b"x")
    with pytest.raises(ValueError, match="layer_stats.csv"):
        artifacts.collect(out, tmp_path / "dest")


def test_collect_removes_stale_files_in_collected_subtrees(tmp_path):
    out, dest = tmp_path / "outputs", tmp_path / "dest"
    (out / "results").mkdir(parents=True)
    (out / "results/gnn.csv").write_bytes(b"new")
    (dest / "results").mkdir(parents=True)
    (dest / "results/stale.csv").write_bytes(b"old")
    (dest / "nodes.csv").write_bytes(b"step02")          # top-level files written directly by step 02 are kept
    artifacts.collect(out, dest)
    assert not (dest / "results/stale.csv").exists()
    assert (dest / "results/gnn.csv").read_bytes() == b"new"
    assert (dest / "nodes.csv").exists()


def test_collect_does_not_cover_the_one_off_cross_check():
    assert not any(p.startswith("features_xcheck") for p, _ in artifacts.PATTERNS)
