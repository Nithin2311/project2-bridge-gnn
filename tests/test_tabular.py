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


def test_mlp_grouped_stopping_is_reproducible(monkeypatch):
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(40), 3)
    y = groups % 4 == 0
    X = rng.normal(size=(len(groups), 4))
    params = {"hidden_layer_sizes": (8,), "alpha": 0.001, "max_iter": 12, "early_stopping": True}
    monkeypatch.setattr(tabular, "grid", lambda _: [params])
    original = tabular.GroupShuffleSplit.split

    def disjoint(self, X, y=None, groups=None):
        for fit, valid in original(self, X, y, groups):
            assert not set(groups[fit]).intersection(groups[valid])
            yield fit, valid

    monkeypatch.setattr(tabular.GroupShuffleSplit, "split", disjoint)
    a = tabular.select("mlp", X, y, groups, 0)
    b = tabular.select("mlp", X, y, groups, 0)
    assert a[0] == b[0] and a[1] == b[1] and a[2].equals(b[2])
    assert 1 <= a[0]["refit_epochs"] <= params["max_iter"]


def test_class_weighted_models_are_prior_corrected():
    """Balanced class weights shift logits by log(n_neg/n_pos); predictions must be put back on the prevalence scale."""
    import numpy as np

    from bridgegnn import tabular
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 4))
    y = rng.uniform(size=n) < 0.08
    est, _, _ = tabular._fit("lr", {"C": 1.0, "class_weight": "balanced", "max_iter": 500}, 0, X, y, np.arange(n))
    p = tabular._probability(est, X)
    assert abs(p.mean() - y.mean()) < 0.03, f"mean prediction {p.mean():.3f} vs prevalence {y.mean():.3f}"
    est_h, _, _ = tabular._fit("hgb", {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 50, "early_stopping": False, "class_weight": "balanced"}, 0, X, y, np.arange(n))
    assert abs(tabular._probability(est_h, X).mean() - y.mean()) < 0.06


def test_hgb_grid_tunes_boosting_rounds():
    from bridgegnn import tabular
    iters = {g["max_iter"] for g in tabular.grid("hgb")}
    assert len(iters) >= 3 and 300 in iters and min(iters) <= 50
