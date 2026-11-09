from datetime import date

import networkx as nx
import polars as pl
import pytest

from bridgegnn import config, nodefeat, rebuild

P1_CFG_PATH = config.p1_repo(config.load()) / "config" / "option_a.yaml"
pytestmark = pytest.mark.skipif(not P1_CFG_PATH.exists(), reason="Project 1 config not present")


@pytest.fixture(scope="module")
def p1_cfg():
    return config.p1_config(config.load())


def test_restricted_table_filters_and_asserts(synth):
    pc = rebuild.restricted_patient_category(synth.codes_long, synth.split, ["train", "val"], "2015-10-01", "2020-01-01")
    assert pc.columns == ["person_id", "category", "first_date"]
    assert (pc["first_date"] < date(2020, 1, 1)).all()
    test_ids = set(synth.split.filter(pl.col("split") == "test")["person_id"])
    assert not (set(pc["person_id"]) & test_ids)
    with pytest.raises(AssertionError):
        rebuild._check_table(pc.with_columns(pl.lit(date(2021, 1, 1)).alias("first_date")), test_ids, "2020-01-01")
    with pytest.raises(AssertionError):
        rebuild._check_table(pc.with_columns(pl.lit(next(iter(test_ids))).cast(pl.Int64).alias("person_id")), test_ids, "2020-01-01")


def test_build_graph_cutoff_has_fewer_or_equal_edges(synth, p1_cfg):
    nodes = synth.nodes.with_columns(pl.lit(False).alias("flagged"))
    pc_all = rebuild.restricted_patient_category(synth.codes_long, synth.split, ["train", "val"], "2015-10-01", "2030-01-01")
    pc_cut = rebuild.restricted_patient_category(synth.codes_long, synth.split, ["train", "val"], "2015-10-01", "2019-01-01")
    loose = {**p1_cfg, "edges": {**p1_cfg["edges"], "min_shared_patients": 2, "min_odds_ratio": 1.0, "alpha": 0.5}}
    g_all, e_all = rebuild.build_graph(pc_all, nodes, loose)
    g_cut, _e_cut = rebuild.build_graph(pc_cut, nodes, loose)
    assert set(g_all.nodes) == set(nodes["category"]) == set(g_cut.nodes)
    assert g_cut.number_of_edges() <= g_all.number_of_edges()
    assert {"u", "v", "phi", "kept", "p_fisher"} <= set(e_all.columns)
    assert all("prevalence" in d for _, d in g_all.nodes(data=True))


def test_cores_bridges_and_node_features(synth, p1_cfg, node_feats):
    g = synth.graph
    anchors = {t: sorted(c)[:1] for t, c in synth.targets["core"].items()}
    partition, target_defs, bridges = rebuild.cores_and_bridges(g, p1_cfg, anchors, n_seeds=4, n_null=2)
    assert set(partition) == set(g.nodes)
    assert {"target", "category", "in_core"} <= set(target_defs.columns)
    assert {"pair", "category", "bridge_betweenness", "bridge_pass"} <= set(bridges.columns)
    nf = rebuild.node_features(g, partition, bridges, node_feats)
    assert set(nodefeat.STRUCT_COLS) | {"category", "community_seed0", "bridge_survivor", "chapter"} <= set(nf.columns)
    _idx, X, _groups = nodefeat.build(nf, bridges, list(anchors))
    assert X.shape[0] == g.number_of_nodes()


def test_compare_reports_overlap(synth):
    g = synth.graph
    h = nx.Graph(); h.add_nodes_from(g.nodes); h.add_edges_from(list(g.edges())[: g.number_of_edges() // 2])
    part = {v: i % 3 for i, v in enumerate(g.nodes)}
    ba = pl.DataFrame({"category": ["A00", "B07"], "bridge_pass": [True, True]})
    bb = pl.DataFrame({"category": ["A00", "C14"], "bridge_pass": [True, True]})
    out = rebuild.compare(h, g, part, part, bb, ba)
    expected = h.number_of_edges() / g.number_of_edges()
    assert abs(out["edge_jaccard"] - expected) < 1e-9 and out["ari"] == 1.0 and abs(out["bridge_jaccard"] - 1 / 3) < 1e-9
