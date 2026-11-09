import numpy as np
import polars as pl
import pytest

from bridgegnn.eval import metrics


def test_bootstrap_perfect_classifier_is_one():
    y = np.array([0, 1] * 50); p = y.astype(float); pid = np.arange(100)
    point, lo, hi = metrics.bootstrap_ci(y, p, pid, metrics.auroc, n=50, seed=0)
    assert (point, lo, hi) == (1.0, 1.0, 1.0)


def test_bootstrap_resamples_whole_patients():
    y = np.array([0, 1, 0, 1]); p = np.array([0.1, 0.9, 0.2, 0.8]); pid = np.array([1, 1, 2, 2])
    seen = []

    def stat(yy, pp):
        seen.append(len(yy))
        return 1.0
    metrics.bootstrap_ci(y, p, pid, stat, n=20, seed=0)
    assert all(k % 2 == 0 for k in seen[1:])   # every resample holds complete patients (2 rows each)


def test_calibration_recovers_slope_one_on_calibrated_probs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.5, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    c = metrics.calibration(y, p)
    assert abs(c["slope"] - 1) < 0.1 and abs(c["intercept"]) < 0.15 and 0 < c["brier"] < 0.25
    rel = metrics.reliability(y, p, bins=10)
    assert rel.height == 10 and set(rel.columns) == {"bin", "p_mean", "y_rate", "n"}


def test_f1_threshold_and_confusion():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.2, 0.6, 0.55, 0.7, 0.9])
    thr = metrics.f1_threshold(y, p)
    c = metrics.confusion(y, p, thr)
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 6 and c["f1"] >= 0.8
    with pytest.raises(ValueError):
        metrics.f1_threshold(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        metrics.f1_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


def test_stratified_hides_small_strata():
    df = pl.DataFrame({"y": [1, 0] * 20 + [1, 0, 0, 0, 0] * 4, "p": np.linspace(0, 1, 60).tolist(),
                       "person_id": list(range(60)), "sex": ["female"] * 40 + ["male"] * 20})
    out = metrics.stratified(df, ["sex"], min_pos=10)
    f = out.filter(pl.col("level") == "female").row(0, named=True)
    m = out.filter(pl.col("level") == "male").row(0, named=True)
    assert f["n"] == 40 and f["pos"] == 20 and f["auroc"] is not None
    assert m["n"] == 20 and m["pos"] == 4 and m["auroc"] is None and m["auprc"] is None


def test_paired_delta_identical_is_zero():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=200) < 0.2).astype(int); p = rng.uniform(size=200); pid = np.arange(200)
    d = metrics.paired_delta(y, p, p.copy(), pid, n=30, seed=0)
    assert d["d_auroc"] == 0 and d["d_auroc_lo"] == 0 and d["d_auroc_hi"] == 0 and d["d_auprc"] == 0
