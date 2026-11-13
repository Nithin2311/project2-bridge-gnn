import json

import numpy as np
import polars as pl
import pytest

from bridgegnn import contract, features, inputs, rows


@pytest.fixture
def feature_inputs(synth, node_feats, cfg):
    labels = inputs.task_labels(synth.labels, cfg)
    masks = rows.mask_sets(synth.target_defs, cfg)
    codes = rows.build_row_codes(labels.select("person_id", "index_date").unique(),
                                 synth.codes_long, synth.nodes, masks, cfg)
    task = rows.build_rows(labels, codes, synth.visits, synth.demo, synth.codes_long, cfg)
    nt = features.node_table(node_feats, synth.bridges, contract.TARGETS)
    return task, codes, nt


def test_masking_and_vocabulary(feature_inputs, cfg):
    task, codes, nt = feature_inputs
    target = "cardiometabolic"
    cm, vocab = features.tabular_for_target(task, codes, nt, target, cfg, None)
    mh, _ = features.tabular_for_target(task, codes, nt, "mental_health", cfg, None)
    masked = codes.filter(pl.col("masked_cardiometabolic") & pl.col("in_main_graph"))
    categories = masked["category"].unique().to_list()
    assert categories and not set(categories).intersection(vocab)
    assert all(cm[f"c_{c}"].sum() == 0 for c in categories)
    assert any(mh[f"c_{c}"].sum() > 0 for c in categories)
    # a category present in development rows but absent from test rows must still be a (zero-filled) column in test
    test_rows = cm.filter(pl.col("split") == "test"); dev_rows = cm.filter(pl.col("split") != "test")
    dev_only = [c for c in cm.columns if c.startswith("all_") and dev_rows[c].sum() > 0 and test_rows[c].sum() == 0]
    assert dev_only, "fixture should contain at least one development-only all_ column"
    assert all(c in test_rows.columns for c in dev_only)
    test_keys = cm.filter(pl.col("split") == "test").select("person_id", "index_date")
    only_test = codes.join(test_keys, on=["person_id", "index_date"], how="semi").with_columns(
        pl.lit("Z99").alias("category"), pl.lit(False).alias("in_main_graph"),
        pl.lit(False).alias("masked_cardiometabolic"),
    ).unique(["person_id", "index_date", "category"])
    _, changed_vocab = features.tabular_for_target(task, pl.concat([codes, only_test]), nt, target, cfg, None)
    assert vocab == changed_vocab
    with pytest.raises(AssertionError):
        features.fit_row_scaler(cm, ["demo_age"], cm["split"].is_not_null())
    contract.validate(cm, "tabular")


def test_empty_pool_and_artifacts(feature_inputs, cfg, tmp_path):
    task, codes, nt = feature_inputs
    codes = codes.with_columns(pl.lit(True).alias("masked_mental_health"))
    tab, vocab = features.tabular_for_target(task, codes, nt, "mental_health", cfg, None)
    assert not vocab
    assert not tab.select("util_n_codes", "util_n_rows", "bridge_n_flag", "pool_mean_degree").to_numpy().any()
    features.write_node_artifacts(nt, tmp_path)
    groups = json.loads((tmp_path / "feature_groups.json").read_text())
    x = np.load(tmp_path / "node_x.npy")
    assert x.shape == (nt.height, len(groups["node_columns"]))
    assert x.dtype == np.float32 and np.isfinite(x).all()


def test_bridge_scores(synth, node_feats):
    target = "mental_health"
    nt = features.node_table(node_feats, synth.bridges, contract.TARGETS)
    observed = synth.bridges.filter(pl.col("pair").str.ends_with(target)).group_by("category").agg(
        pl.col("bridge_betweenness").max().alias("expected"))
    assert observed.join(nt, on="category").select(
        (pl.col("expected") == pl.col(f"bridge_score_{target}")).all()).item()


def test_build_synthetic_end_to_end(cfg, tmp_path):
    import json

    import numpy as np
    import polars as pl

    from bridgegnn import features, synth
    features.build_synthetic(cfg, tmp_path, verbose=False)
    targets = cfg["task"]["primary_targets"] + cfg["task"]["secondary_targets"]
    for t in targets:
        tabs = {s: pl.read_parquet(tmp_path / f"tabular_{t}_{s}.parquet") for s in ("train", "val", "test")}
        assert tabs["train"].columns == tabs["val"].columns == tabs["test"].columns
        dev = pl.concat([tabs["train"], tabs["val"]])
        all_cols = [c for c in dev.columns if c.startswith("all_")]
        dev_only = [c for c in all_cols if tabs["test"][c].sum() == 0 and dev[c].sum() > 0]
        assert all_cols and all(tabs["test"][c].sum() == 0 for c in dev_only)
        # bridge row features come from the RAW (unstandardised) per-target scores
        assert (dev["bridge_max_score"] >= 0).all() and (dev["bridge_n_flag"] == dev["bridge_n_flag"].round(0)).all()
    data = synth.make(0)
    X = np.load(tmp_path / "node_x.npy")
    groups = json.loads((tmp_path / "feature_groups.json").read_text())
    assert X.shape[0] == data.nodes.height and len(groups["node_columns"]) == X.shape[1]
    assert set(groups) >= {"STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY", "node_columns", "CODE", "ALL", "DEMO", "UTIL", "TEMP", "POOL"}
    assert groups["CODE"] and groups["POOL"]
