import numpy as np
import polars as pl

from bridgegnn import nodefeat
from bridgegnn.contract import TARGETS


def test_build_shapes_and_groups(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    assert X.shape[0] == idx.height == synth.nodes.height and X.dtype == np.float32
    assert set(groups) == {"STRUCT", "BRIDGE", "CHAPTER", "COMMUNITY"}
    assert sum(len(v) for v in groups.values()) == X.shape[1]
    assert idx["category"].to_list() == sorted(idx["category"].to_list()) and idx["idx"].to_list() == list(range(idx.height))
    assert abs(float(X[:, 0].mean())) < 1e-5 and abs(float(X[:, 0].std()) - 1) < 1e-3
    assert np.isfinite(X).all()


def test_bridge_flag_per_target_is_union_over_pairs(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    col = X.shape[1] - X.shape[1]  # placeholder to keep flake quiet
    names = groups["STRUCT"] + groups["BRIDGE"] + groups["CHAPTER"] + groups["COMMUNITY"]
    j = names.index("bridge_flag_mental_health")
    flagged = set(idx.filter(pl.Series(X[:, j] > 0))["category"])
    expected = set(synth.bridges.filter(pl.col("bridge_pass") & pl.col("pair").str.contains("mental_health"))["category"]) & set(node_feats["category"])
    assert flagged == expected and col == 0


def test_bridge_score_is_max_over_pairs(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS, standardise=False)
    names = groups["STRUCT"] + groups["BRIDGE"] + groups["CHAPTER"] + groups["COMMUNITY"]
    j = names.index("bridge_score_mental_health")
    cat = idx["category"][3]
    exp = synth.bridges.filter((pl.col("category") == cat) & pl.col("pair").str.contains("mental_health"))["bridge_betweenness"].max()
    assert abs(float(X[3, j]) - float(exp or 0.0)) < 1e-9


def test_bflag_codes_matches_matrix(node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    names = groups["STRUCT"] + groups["BRIDGE"] + groups["CHAPTER"] + groups["COMMUNITY"]
    j = names.index("bridge_flag_cardiometabolic")
    assert nodefeat.bflag_codes(node_feats, synth.bridges, "cardiometabolic") == set(idx.filter(pl.Series(X[:, j] > 0))["category"])


def test_community_onehot_groups_small_communities(node_feats, synth):
    _, _, groups = nodefeat.build(node_feats, synth.bridges, TARGETS, min_community_size=5)
    assert "community_other" in groups["COMMUNITY"]


def test_write_read_roundtrip(tmp_path, node_feats, synth):
    idx, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    nodefeat.write(idx, X, groups, tmp_path)
    idx2, X2, groups2 = nodefeat.read(tmp_path)
    assert idx2.equals(idx) and np.allclose(X, X2) and groups2 == groups
