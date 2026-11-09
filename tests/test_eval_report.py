from datetime import date

import numpy as np
import polars as pl
import pytest

from bridgegnn import contract
from bridgegnn.eval import report

CFG = {"eval": {"n_bootstrap": 20, "seed": 0, "min_stratum_pos": 3, "calib_bins": 5, "baselines_for_pairing": ["B-hgb_all"],
                "paired_runs": [["G-sage", "B-hgb_all"]]},
       "task": {"primary_targets": ["mental_health"]}}


def make_preds(run_id, model, seeds, n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.random.default_rng(100).uniform(size=n) < 0.2   # labels are a property of the rows, shared by every run
    frames = []
    for s in seeds:
        pid = np.repeat(np.arange(n // 2), 2)
        split = np.where(pid % 3 == 0, "test", np.where(pid % 3 == 1, "train", "val"))
        p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, size=n), 0.01, 0.99)
        frames.append(pl.DataFrame({"run_id": [run_id] * n, "model": [model] * n, "target": ["mental_health"] * n,
                                    "seed": pl.Series([s] * n, dtype=pl.Int32), "person_id": pl.Series(pid, dtype=pl.Int64),
                                    "index_date": pl.Series([date(2018 + (i % 4), 1, 1) for i in range(n)], dtype=pl.Date),
                                    "split": split.tolist(), "y": y.tolist(), "p": p.tolist()}))
    return pl.concat(frames)


def make_rows(preds):
    r = preds.select("person_id", "index_date", "target").unique()
    rng = np.random.default_rng(3)
    return r.with_columns(pl.Series("sex", rng.choice(["female", "male"], size=r.height).tolist()),
                          pl.Series("age_at_index", rng.uniform(30, 90, size=r.height)),
                          pl.Series("n_codes_lb1095", rng.integers(0, 20, size=r.height).tolist()).cast(pl.Int32),
                          pl.col("index_date").dt.year().cast(pl.Int32).alias("index_year"))


@pytest.fixture(scope="module")
def preds_two_runs():
    p = pl.concat([make_preds("G-sage", "sage", [0, 1], seed=0), make_preds("B-hgb_all", "hgb_all", [0, 1], seed=1)])
    contract.validate(p, "predictions")
    return p


def test_summarise_produces_all_tables(preds_two_runs):
    rows = make_rows(preds_two_runs)
    out = report.summarise(preds_two_runs, rows, CFG)
    assert set(out) == {"runs", "runs_agg", "reliability", "strata", "paired"}
    runs = out["runs"]
    assert runs.height == 4 and {"run_id", "target", "seed", "auroc", "auprc", "brier", "slope", "citl", "f1", "threshold", "threshold_source",
                                 "pred_pos_rate", "degenerate_threshold", "tp", "fp", "fn", "tn", "n", "pos"} <= set(runs.columns)
    assert set(runs["threshold_source"]) == {"dev_oof"}
    agg = out["runs_agg"]
    assert agg.height == 2 and {"auroc_mean", "auroc_sd", "auroc_seed", "auroc_ci_seed", "auroc_lo", "auroc_hi", "auprc_mean", "auprc_sd",
                                "auprc_seed", "auprc_lo", "auprc_hi", "brier_mean", "brier_sd", "f1_mean", "n_seeds", "ci_seed"} <= set(agg.columns)
    assert (agg["ci_seed"] == 0).all()
    assert set(out["strata"]["stratum"]) == {"sex", "age_band", "size_bin", "index_year"}
    assert out["reliability"].filter(pl.col("run_id") == "G-sage").height == 5
    paired = out["paired"]
    # one pairing from baselines_for_pairing and the same pairing again from paired_runs -> deduplicated
    assert paired.height == 2 and set(paired["baseline"]) == {"B-hgb_all"} and {"d_auroc", "d_auroc_lo", "d_auroc_hi", "d_auprc"} <= set(paired.columns)


def test_paired_skipped_without_baseline(preds_two_runs):
    only = preds_two_runs.filter(pl.col("run_id") == "G-sage")
    out = report.summarise(only, make_rows(only), CFG)
    assert out["paired"].height == 0


def test_threshold_uses_dev_rows_only(preds_two_runs):
    one = preds_two_runs.filter((pl.col("run_id") == "G-sage") & (pl.col("seed") == 0))
    thr = report.dev_threshold(one)
    dev = one.filter(pl.col("split") != "test")
    from bridgegnn.eval import metrics
    assert thr == metrics.f1_threshold(dev["y"].to_numpy(), dev["p"].to_numpy())
    with pytest.raises(ValueError):
        report.dev_threshold(one.filter(pl.col("split") == "test"))


def test_load_predictions_reads_every_file(tmp_path, preds_two_runs):
    for (rid, s), sub in preds_two_runs.group_by("run_id", "seed"):
        contract.write(sub, tmp_path / f"{rid}_mental_health_seed{s}.parquet", "predictions")
    loaded = report.load_predictions(tmp_path)
    assert loaded.height == preds_two_runs.height


def test_summarise_flags_missing_dev_rows_instead_of_silent_fallback(preds_two_runs):
    one = preds_two_runs.filter((pl.col("run_id") == "G-sage") & (pl.col("split") == "test"))
    out = report.summarise(one, make_rows(one), CFG)
    r = out["runs"]
    assert set(r["threshold_source"]) == {"fallback"} and r["f1"].is_null().all()


def test_to_latex_escapes_underscores_in_cells():
    df = pl.DataFrame({"target": ["cancer_hematologic"], "run_id": ["B-hgb_all"], "v": [0.5]})
    tex = report.to_latex(df, ["target", "run_id", "v"], "cap", "tab:x")
    body = tex.split("\\midrule")[1]
    import re
    assert not re.search(r"(?<!\\)_", body)
    assert "cancer\\_hematologic" in body and "B-hgb\\_all" in body


def test_paired_runs_pairing_between_two_gnn_runs():
    a = make_preds("B5-P", "sage", [0], seed=5); b = make_preds("B5-U", "sage", [0], seed=6)
    cfg = {"eval": {**CFG["eval"], "baselines_for_pairing": [], "paired_runs": [["B5-P", "B5-U"]]}, "task": CFG["task"]}
    out = report.summarise(pl.concat([a, b]), make_rows(pl.concat([a, b])), cfg)
    assert out["paired"].height == 1 and out["paired"]["baseline"][0] == "B5-U"
