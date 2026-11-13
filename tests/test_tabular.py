import numpy as np
import polars as pl
import pytest

from bridgegnn import contract, tabular


def test_feature_columns():
    cols = ["c_X99", "all_Z99", "demo_age", "util_n_rows", "bridge_n_flag", "temp_share_recent",
            "pool_mean_degree", "pool_max_degree"]
    assert tabular.feature_columns("lr", cols) == ["c_X99", "demo_age", "util_n_rows"]
    assert tabular.feature_columns("hgb_all", cols) == ["all_Z99", "demo_age", "util_n_rows"]
    assert tabular.feature_columns("mlp_pool", cols) == [c for c in cols if c != "all_Z99"]


def test_grouped_selection_handles_single_class(monkeypatch):
    groups = np.repeat(np.arange(20), 3)
    y = np.zeros(len(groups), dtype=bool)
    y[[1, 23]] = True
    original = tabular.GroupKFold.split

    def checked(self, X, y=None, groups=None):
        for fit, valid in original(self, X, y, groups):
            assert not set(groups[fit]).intersection(groups[valid])
            yield fit, valid

    monkeypatch.setattr(tabular.GroupKFold, "split", checked)
    with pytest.warns(UserWarning, match="single class"):
        _, score, oof = tabular.select("lr", groups[:, None], y, groups, 0)
    assert np.isfinite(score)
    assert oof.height == len(groups) and oof["p"].is_finite().all()


def test_test_patients_only_used_for_predict(cfg, monkeypatch):
    from datetime import date

    tables = {}
    for split, start in [("train", 0), ("val", 20), ("test", 40)]:
        ids = list(range(start, start + 20))
        tables[split] = pl.DataFrame({
            "person_id": ids, "index_date": [date(2020, 1, 1)] * 20,
            "target": ["mental_health"] * 20, "split": [split] * 20,
            "positive": [i % 7 == 0 for i in ids], "demo_age": ids,
        })
    original = tabular.make

    def checked(model, params, seed):
        est = original(model, params, seed)
        fit = est.fit

        def checked_fit(X, y):
            assert (X[:, 0] < 40).all()
            return fit(X, y)

        est.fit = checked_fit
        return est

    monkeypatch.setattr(tabular, "make", checked)
    predictions, results = tabular.run("lr", "mental_health", tables, cfg, seed=1)
    contract.validate(predictions, "predictions")
    assert results["n_test_rows"] == 20
    monkeypatch.setattr(tabular, "make", original)
    preds, _ = tabular.run("prevalence", "mental_health", tables, cfg, seed=0)
    assert preds.filter(pl.col("split") == "test")["p"].n_unique() == 1
