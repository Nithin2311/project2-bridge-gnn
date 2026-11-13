import polars as pl

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
