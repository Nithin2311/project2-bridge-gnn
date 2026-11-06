import polars as pl
import pytest

from bridgegnn import config, inputs

REAL_P1 = config.p1_outputs(config.load())


def test_task_labels_filters_to_core_12_uncensored(synth):
    cfg = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True}}
    out = inputs.task_labels(synth.labels, cfg)
    assert "censored" not in out.columns and "definition" not in out.columns and "horizon_months" not in out.columns
    raw = synth.labels.filter((pl.col("definition") == "core") & (pl.col("horizon_months") == 12) & ~pl.col("censored"))
    assert out.height == raw.height
    assert out.height > 0


def test_check_reports_missing(tmp_path):
    assert set(inputs.check(tmp_path)) == set(inputs.P1_FILES)


def test_class_balance_has_all_row(synth):
    cfg = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True}}
    bal = inputs.class_balance(inputs.task_labels(synth.labels, cfg))
    assert set(bal.columns) == {"target", "split", "rows", "patients", "positives", "positive_rate"}
    assert "all" in set(bal["split"])


@pytest.mark.skipif(not REAL_P1.exists(), reason="Project 1 outputs not present")
def test_real_counts_match_label_counts():
    labels = inputs.task_labels(pl.read_parquet(REAL_P1 / "p2_labels.parquet"), config.load())
    bal = inputs.class_balance(labels)
    cm = bal.filter((pl.col("target") == "cardiometabolic") & (pl.col("split") == "all")).row(0, named=True)
    assert cm["positives"] == 163 and cm["rows"] == 1863 - 182
