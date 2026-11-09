import json

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from bridgegnn import contract, inputs, nodefeat, rows
from bridgegnn.contract import TARGETS
from bridgegnn.gnn import data, train

CFG = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True},
       "windows": {"lookback_days": [365, 1095, 0], "primary_lookback": 1095, "era_start": "2015-10-01",
                   "recent_days": 90, "visit_window_days": 365},
       "masking": {"input_exclusion": ["core", "chapter_block"]},
       "split": {"dev": ["train", "val"], "test": ["test"], "cv_folds": 3, "group_by": "person_id"},
       "gnn": {"hidden": 16, "depth": 2, "dropout": 0.1, "heads": 4, "agg": "mean", "batch_size": 8,
               "lr_grid": [1e-3], "weight_decay_grid": [0.0], "max_epochs": 3, "patience": 2, "embedding_dim": 8,
               "selection_seed": 0, "device": "cpu"}}
ALL = ["CODE", "DEMO", "UTIL", "TEMP", "STRUCT", "BRIDGE"]


@pytest.fixture(scope="module")
def setup(synth, node_feats):
    labels = inputs.task_labels(synth.labels, CFG)
    idx_rows = labels.select("person_id", "index_date").unique()
    masks = rows.mask_sets(synth.target_defs, CFG)
    rc = rows.build_row_codes(idx_rows, synth.codes_long, synth.nodes, masks, CFG)
    r = rows.build_rows(labels, rc, synth.visits, synth.demo, synth.codes_long, CFG)
    node_index, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    opts = data.Options("U", ALL, 1095, False, True, set(), False, False)
    rows_t = r.filter(pl.col("target") == "mental_health")
    exs = data.build_examples(rows_t, rc, "mental_health", node_index, X, groups, synth.graph, opts, CFG)
    dev = [e for e in exs if e.split != "test"]
    test = [e for e in exs if e.split == "test"]
    names = data.covariate_names(ALL)
    kwargs = {"arch": "sage", "in_dim": dev[0].x.shape[1], "cov_dim": len(names), "hidden": 16, "depth": 2, "dropout": 0.1, "heads": 4,
                  "agg": "mean", "n_nodes": node_index.height, "embedding_dim": 8, "use_embedding": True}
    return {"dev": dev, "test": test, "names": names, "kwargs": kwargs}


def test_group_folds_never_split_a_patient(setup):
    dev = setup["dev"]
    folds = train.group_folds(dev, 3, 0)
    assert len(folds) == 3
    for tr, va in folds:
        assert not ({dev[i].person_id for i in tr} & {dev[i].person_id for i in va})
        assert len(tr) + len(va) == len(dev)


def test_group_folds_seed_changes_assignment(setup):
    a = train.group_folds(setup["dev"], 3, 0)
    b = train.group_folds(setup["dev"], 3, 1)
    assert any(set(a[i][1].tolist()) != set(b[i][1].tolist()) for i in range(3))


def test_fit_returns_finite_predictions_and_curves(setup):
    dev = setup["dev"]
    m, best, curves = train.fit(dev[:30], dev[30:40], {"lr": 1e-3, "weight_decay": 0.0}, 0, CFG["gnn"], setup["kwargs"])
    p = train.predict(m, dev[30:40])
    assert p.shape == (10,) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert len(curves["train_loss"]) == len(curves["val_auprc"]) >= 1 and 1 <= best <= len(curves["train_loss"])


def test_fit_is_seeded(setup):
    dev = setup["dev"]
    m1, _, _ = train.fit(dev[:30], [], {"lr": 1e-3, "weight_decay": 0.0, "epochs": 2}, 0, CFG["gnn"], setup["kwargs"])
    m2, _, _ = train.fit(dev[:30], [], {"lr": 1e-3, "weight_decay": 0.0, "epochs": 2}, 0, CFG["gnn"], setup["kwargs"])
    assert np.allclose(train.predict(m1, dev[30:40]), train.predict(m2, dev[30:40]))


def test_curves_include_val_loss_and_predictions_are_prior_corrected(setup):
    dev = setup["dev"]
    m, _best, curves = train.fit(dev[:30], dev[30:40], {"lr": 1e-3, "weight_decay": 0.0}, 0, CFG["gnn"], setup["kwargs"])
    assert len(curves["val_loss"]) == len(curves["val_auprc"]) and all(np.isfinite(curves["val_loss"]))
    assert hasattr(m, "logit_shift") and m.logit_shift > 0
    p = train.predict(m, dev[:30])
    raw = train.predict(m, dev[:30], prior_correct=False)
    assert np.all(p <= raw + 1e-9) and np.corrcoef(p, raw)[0, 1] > 0.99


def test_select_logs_every_grid_point(setup):
    grid = [{"lr": 1e-3, "weight_decay": 0.0}, {"lr": 3e-4, "weight_decay": 0.0}]
    _best, _score, _oof, _epochs, _fold_curves, grid_log = train.select(setup["dev"], grid, CFG, setup["kwargs"], 0, setup["names"])
    assert len(grid_log) == 2 and all({"params", "fold_scores", "fold_epochs", "mean_score"} <= set(g) for g in grid_log)


def test_refit_trains_fixed_epochs(setup):
    _m, best, curves = train.fit(setup["dev"], [], {"lr": 1e-3, "weight_decay": 0.0, "epochs": 2}, 0, CFG["gnn"], setup["kwargs"])
    assert best == 2 and len(curves["train_loss"]) == 2 and curves["val_auprc"] == []


def test_select_returns_oof_for_every_dev_row(setup):
    dev = setup["dev"]
    best, score, oof, epochs, fold_curves, _grid_log = train.select(dev, [{"lr": 1e-3, "weight_decay": 0.0}], CFG, setup["kwargs"], 0, setup["names"])
    assert oof.shape == (len(dev),) and np.isfinite(oof).all() and epochs >= 1 and len(fold_curves) == 3
    assert best == {"lr": 1e-3, "weight_decay": 0.0} and (score is None or 0 <= score <= 1)


def test_select_skips_single_class_fold(setup, monkeypatch):
    dev = [data.Example(e.x, e.edge_index, e.edge_weight, e.node_ids, e.cov, 0, e.person_id, e.index_date, e.split) for e in setup["dev"]]
    dev[0] = data.Example(dev[0].x, dev[0].edge_index, dev[0].edge_weight, dev[0].node_ids, dev[0].cov, 1, dev[0].person_id, dev[0].index_date, dev[0].split)
    with pytest.warns(UserWarning):
        _best, score, _oof, epochs, _, _ = train.select(dev, [{"lr": 1e-3, "weight_decay": 0.0}], CFG, setup["kwargs"], 0, setup["names"])
    assert epochs >= 1 and (score is None or np.isfinite(score))


def test_run_writes_contract_files_and_replaces_rows(tmp_path, setup):
    flags = {"arch": "sage", "layer": "U", "features": ",".join(ALL)}
    res = train.run("T-sage", "mental_health", setup["dev"], setup["test"], [{"lr": 1e-3, "weight_decay": 0.0}], [0, 1],
                    CFG, setup["kwargs"], flags, tmp_path, setup["names"], family="gnn")
    assert len(res) == 2 and {r["seed"] for r in res} == {0, 1}
    preds = contract.read(tmp_path / "predictions" / "T-sage_mental_health_seed0.parquet", "predictions")
    assert preds.filter(pl.col("split") == "test").height == len(setup["test"])
    assert preds.filter(pl.col("split") != "test").height == len(setup["dev"])
    results = contract.read(tmp_path / "results" / "gnn.csv", "results")
    assert results.height == 2 and set(results["run_id"]) == {"T-sage"}
    curves = json.loads((tmp_path / "logs" / "06_T-sage_mental_health_seed0.json").read_text())
    assert "refit" in curves and "selection" in curves and "grid" in curves and "val_loss" in curves["refit"]
    train.run("T-sage", "mental_health", setup["dev"], setup["test"], [{"lr": 1e-3, "weight_decay": 0.0}], [0],
              CFG, setup["kwargs"], flags, tmp_path, setup["names"], family="gnn")
    assert contract.read(tmp_path / "results" / "gnn.csv", "results").height == 2
    r0 = results.row(0, named=True)
    assert r0["n_test_rows"] == len(setup["test"]) and json.loads(r0["params_json"])["epochs"] >= 1


def test_run_handles_virtual_only_rows(tmp_path, setup):
    def empty(e):
        x = np.zeros((1, e.x.shape[1]), np.float32); x[0, -2] = 1
        return data.Example(x, np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.array([-1]), e.cov, e.y, e.person_id, e.index_date, e.split)
    dev = [empty(e) for e in setup["dev"]]; test = [empty(e) for e in setup["test"]]
    res = train.run("T-empty", "mental_health", dev, test, [{"lr": 1e-3, "weight_decay": 0.0}], [0], CFG, setup["kwargs"],
                    {"arch": "sage"}, tmp_path, setup["names"])
    assert len(res) == 1 and np.isfinite(res[0]["test_brier"])
