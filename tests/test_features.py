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
    assert all(cm.filter(pl.col("split") == s).schema == cm.schema for s in ("train", "val", "test"))
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
