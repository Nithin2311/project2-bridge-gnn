from dataclasses import replace

import networkx as nx
import numpy as np
import polars as pl
import pytest

from bridgegnn import inputs, nodefeat, rows
from bridgegnn.contract import TARGETS
from bridgegnn.gnn import data

CFG = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True},
       "windows": {"lookback_days": [365, 1095, 0], "primary_lookback": 1095, "era_start": "2015-10-01",
                   "recent_days": 90, "visit_window_days": 365},
       "masking": {"input_exclusion": ["core", "chapter_block"]}}
ALL = ["CODE", "DEMO", "UTIL", "TEMP", "STRUCT", "BRIDGE"]


@pytest.fixture(scope="module")
def built(synth, node_feats):
    labels = inputs.task_labels(synth.labels, CFG)
    idx_rows = labels.select("person_id", "index_date").unique()
    masks = rows.mask_sets(synth.target_defs, CFG)
    rc = rows.build_row_codes(idx_rows, synth.codes_long, synth.nodes, masks, CFG)
    r = rows.build_rows(labels, rc, synth.visits, synth.demo, synth.codes_long, CFG)
    node_index, X, groups = nodefeat.build(node_feats, synth.bridges, TARGETS)
    opts = data.Options(layer="U", features=ALL, lookback=1095, expand=False, weighted=True, drop_codes=set(),
                        directed=False, symmetrise=False)
    rows_t = r.filter(pl.col("target") == "mental_health")
    return dict(rows_t=rows_t, rc=rc, node_index=node_index, X=X, groups=groups, g=synth.graph, opts=opts)


def ex(b, **kw):
    opts = replace(b["opts"], **kw) if kw else b["opts"]
    return data.build_examples(b["rows_t"], b["rc"], "mental_health", b["node_index"], b["X"], b["groups"], b["g"], opts, CFG)


def test_example_count_and_shapes(built):
    exs = ex(built)
    assert len(exs) == built["rows_t"].height
    n_static = sum(len(v) for v in built["groups"].values())
    e = exs[0]
    assert e.x.shape[1] == n_static + 3 + 2 and e.x.dtype == np.float32
    assert e.edge_index.shape[0] == 2 and e.edge_index.dtype == np.int64 and e.edge_weight.shape[0] == e.edge_index.shape[1]
    assert e.cov.shape[0] == len(data.covariate_names(ALL)) and e.y in (0, 1) and e.split in {"train", "val", "test"}
    assert e.node_ids.shape[0] == e.x.shape[0]


def test_masked_codes_never_appear(built):
    masked = set(built["rc"].filter(pl.col("masked_mental_health"))["category"])
    cat_of = dict(zip(built["node_index"]["idx"], built["node_index"]["category"]))
    for e in ex(built):
        assert not ({cat_of[i] for i in e.node_ids if i >= 0} & masked)


def test_dropped_codes_absent_and_util_adjusted(built):
    exs0 = ex(built)
    big = max(exs0, key=lambda e: e.x.shape[0])
    cat_of = dict(zip(built["node_index"]["idx"], built["node_index"]["category"]))
    victim = cat_of[int(big.node_ids[0])]
    exs1 = ex(built, drop_codes={victim})
    e1 = next(e for e in exs1 if e.person_id == big.person_id and e.index_date == big.index_date)
    assert victim not in {cat_of[i] for i in e1.node_ids if i >= 0}
    names = data.covariate_names(ALL)
    k = names.index("util_n_codes_lb1095")
    assert e1.cov[k] == big.cov[k] - 1


def test_code_only_features_leave_only_flags(built):
    e = ex(built, features=["CODE"])[0]
    assert e.x.shape[1] == 2 and e.cov.shape[0] == 0


def test_empty_row_gets_virtual_node(built):
    all_codes = set(built["node_index"]["category"])
    e = ex(built, drop_codes=all_codes)[0]
    assert e.x.shape[0] == 1 and e.node_ids[0] == -1 and e.edge_index.shape[1] == 0 and e.x[0, -2] == 1 and e.x[0, -1] == 0


def test_expand_adds_context_nodes_with_flag(built):
    e0 = ex(built)[0]
    e1 = ex(built, expand=True)[0]
    assert e1.x.shape[0] >= e0.x.shape[0]
    assert int((e1.x[:, -1] == 1).sum()) == e1.x.shape[0] - e0.x.shape[0]
    temp_cols = slice(e1.x.shape[1] - 5, e1.x.shape[1] - 2)
    assert np.all(e1.x[e1.x[:, -1] == 1][:, temp_cols] == 0)


def test_directed_layer_symmetrised_only_for_gcn(built):
    e0 = max(ex(built), key=lambda e: e.x.shape[0])
    cat_of = dict(zip(built["node_index"]["idx"], built["node_index"]["category"]))
    a, b = cat_of[int(e0.node_ids[0])], cat_of[int(e0.node_ids[1])]
    dg = nx.DiGraph()
    dg.add_nodes_from(built["g"].nodes(data=True))
    dg.add_edge(a, b, weight=0.7)
    pick = lambda exs: next(e for e in exs if e.person_id == e0.person_id and e.index_date == e0.index_date)
    d = built | {"g": dg}
    e_dir = pick(data.build_examples(d["rows_t"], d["rc"], "mental_health", d["node_index"], d["X"], d["groups"], dg, replace(d["opts"], layer="D", directed=True), CFG))
    e_sym = pick(data.build_examples(d["rows_t"], d["rc"], "mental_health", d["node_index"], d["X"], d["groups"], dg, replace(d["opts"], layer="D", directed=True, symmetrise=True), CFG))
    assert e_dir.edge_index.shape[1] == 1 and e_sym.edge_index.shape[1] == 2


def test_undirected_edges_appear_in_both_directions_and_weights(built):
    e = max(ex(built), key=lambda e: e.edge_index.shape[1])
    pairs = set(map(tuple, e.edge_index.T.tolist()))
    assert all((v, u) in pairs for u, v in pairs)
    assert not np.all(e.edge_weight == 1.0)
    e_un = next(x for x in ex(built, weighted=False) if x.person_id == e.person_id and x.index_date == e.index_date)
    assert np.all(e_un.edge_weight == 1.0)


def test_cov_scaler_fit_only_on_given_examples(built):
    exs = ex(built)
    dev = [e for e in exs if e.split != "test"]
    names = data.covariate_names(ALL)
    mean, sd = data.fit_cov_scaler(dev, names)
    assert mean.shape == (len(names),) and (sd > 0).all()
    scaled = data.apply_cov_scaler(dev, mean, sd, names)
    k = names.index("demo_age")
    assert abs(float(np.mean([e.cov[k] for e in scaled]))) < 1e-5
    kf = names.index("demo_female")
    assert set(np.unique([e.cov[kf] for e in scaled])) <= {0.0, 1.0}
