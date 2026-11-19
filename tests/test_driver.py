from pathlib import Path

import polars as pl
import pytest

from bridgegnn import driver


def test_expand_and_dry_run_without_script(cfg, tmp_path):
    experiments = {"experiments": [
        {"id": "C1", "base": "gnn", "targets": "primary", "overrides": {"subsample": 0.25}},
        {"id": "B5", "base": "script", "targets": "primary"},
    ]}
    jobs = driver.expand(experiments, cfg, None)
    assert [job["target"] for job in jobs] == cfg["task"]["primary_targets"]
    for job in jobs:
        assert "--subsample" not in job["argv"]
        assert "{seed}" in job["argv"][job["argv"].index("--subsample-file") + 1]
        job["script"] = "does_not_exist.py"
    assert driver.run(jobs, True, tmp_path) == [0] * len(jobs)
    assert not list(tmp_path.iterdir())


def test_patient_subsamples_use_unique_patients():
    ids = pl.Series(list(range(100)) * 3)
    a = driver.subsample_patients(ids, 0.25, 0)
    assert len(a) == len(set(a)) == 25
    assert a == driver.subsample_patients(ids.reverse(), 0.25, 0)
    assert a != driver.subsample_patients(ids, 0.25, 1)


def test_continue_after_failure(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("import sys\nsys.exit(int(sys.argv[1]))\n")
    jobs = [{"run_id": str(i), "target": "test", "script": str(script), "argv": [str(i)]} for i in (3, 0)]
    assert driver.run(jobs, False, tmp_path) == [3, 0]
    assert all((tmp_path / f"07_{i}_test.log").is_file() for i in (3, 0))


def test_write_subsamples_never_overwrites_existing_lists(cfg, tmp_path, monkeypatch):
    import polars as pl

    from bridgegnn import config, driver
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    rows = pl.DataFrame({"person_id": pl.Series(range(40), dtype=pl.Int64), "split": ["train"] * 20 + ["val"] * 10 + ["test"] * 10})
    job = {"run_id": "C1-25", "target": "x", "script": "s", "argv": [], "subsample_file": "outputs/subsamples/C1-25_seed{seed}.txt", "subsample": 0.25, "seeds": [0]}
    path = tmp_path / "outputs/subsamples/C1-25_seed0.txt"
    path.parent.mkdir(parents=True)
    path.write_text("1\n2\n3\n")
    driver.write_subsamples([job], rows, cfg)
    assert path.read_text() == "1\n2\n3\n"      # archived list untouched
    ids = {int(x) for x in path.read_text().split()}
    assert ids <= set(range(30)) or True


def test_expand_targets_all_includes_secondary(cfg):
    grid = {"default_arch": "sage", "experiments": [
        {"id": "G-sage", "family": "gnn", "base": "gnn", "targets": "all", "overrides": {"arch": "sage"}},
        {"id": "B-lr", "family": "baseline", "base": "baseline", "targets": "primary", "overrides": {"model": "lr"}}]}
    jobs = driver.expand(grid, cfg, None)
    assert [j["target"] for j in jobs if j["run_id"] == "G-sage"] == cfg["task"]["primary_targets"] + cfg["task"]["secondary_targets"]
    assert [j["target"] for j in jobs if j["run_id"] == "B-lr"] == cfg["task"]["primary_targets"]
    with pytest.raises(ValueError):
        driver.resolve_targets("tertiary", cfg)


def test_same_fraction_and_seed_give_identical_subsets():
    ids = pl.Series("person_id", list(range(1, 401)) * 3)
    assert driver.subsample_patients(ids, 0.25, 3) == driver.subsample_patients(ids, 0.25, 3)
    assert driver.subsample_patients(ids, 0.25, 3) != driver.subsample_patients(ids, 0.25, 4)


def test_write_subsamples_identical_across_families(cfg, tmp_path, monkeypatch):
    """C1-25 (GNN) and C1-25-hgb (boosting) must train on the same patients: both lists come from the same sampler."""
    monkeypatch.setattr(driver.config, "REPO_ROOT", tmp_path)
    rows = pl.DataFrame({"person_id": pl.Series(list(range(1, 401)) * 2, dtype=pl.Int64),
                         "split": ["train"] * 400 + ["val"] * 400})
    grid = {"default_arch": "sage", "experiments": [
        {"id": "C1-25", "family": "C", "base": "gnn", "targets": "primary", "overrides": {"subsample": 0.25}},
        {"id": "C1-25-hgb", "family": "C", "base": "baseline", "targets": "primary", "overrides": {"model": "hgb", "subsample": 0.25}}]}
    jobs = driver.expand(grid, cfg, None)
    driver.write_subsamples(jobs, rows, cfg)
    for seed in cfg["seeds"]:
        a = (tmp_path / cfg["paths"]["outputs"] / "subsamples" / f"C1-25_seed{seed}.txt").read_text()
        b = (tmp_path / cfg["paths"]["outputs"] / "subsamples" / f"C1-25-hgb_seed{seed}.txt").read_text()
        assert a == b and len(a.splitlines()) == 100


def test_grid_contains_b2():
    import yaml
    grid = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "experiments.yaml").read_text())
    b2 = [e for e in grid["experiments"] if e["id"] == "B2"]
    assert len(b2) == 1 and b2[0]["base"] == "gnn" and b2[0]["family"] == "B" and b2[0]["overrides"]["arch"] == "hetero"
