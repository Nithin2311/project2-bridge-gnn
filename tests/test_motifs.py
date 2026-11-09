from datetime import date

import networkx as nx
import polars as pl

from bridgegnn import motifs


def ff():
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return g


def test_triad_census_counts_feed_forward():
    c = motifs.triad_census(ff())
    assert c["030T"] == 1 and sum(c.values()) == 1


def test_null_census_shape_and_z(synth):
    dg = nx.DiGraph()
    dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:40])
    out = motifs.null_census(dg, n=5, seed=0)
    assert out.height == 16 and {"triad", "observed", "null_mean", "null_sd", "z", "p_emp"} <= set(out.columns)
    assert (out["observed"].sum()) == sum(motifs.triad_census(dg).values())
    zero_sd = out.filter(pl.col("null_sd") == 0)
    assert zero_sd["z"].is_null().all()


def test_null_census_is_seeded(synth):
    dg = nx.DiGraph()
    dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:40])
    a = motifs.null_census(dg, 3, 1)
    b = motifs.null_census(dg, 3, 1)
    assert a.equals(b)


def test_feed_forward_middle_share():
    out = motifs.feed_forward_middle_share(ff(), {"B", "A", "C"})
    get = lambda c: out.filter(pl.col("code") == c).row(0, named=True)
    assert get("B")["n_middle"] == 1 and get("A")["n_middle"] == 0 and get("C")["n_middle"] == 0
    assert abs(get("B")["share_of_ff"] - 1.0) < 1e-9


def test_bridge_chains_requires_order_and_excludes_core_as_a():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=pl.Int64),
                       "category": ["X", "N17", "I50", "X", "N17", "I50", "I50", "N17", "X", "E11", "N17", "I50"],
                       "first_date": [date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1),
                                      date(2018, 1, 1), date(2018, 1, 10), date(2018, 6, 1),   # gap too small on step 1
                                      date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1),    # wrong order
                                      date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1)]})  # a is a core member
    out = motifs.bridge_chains(fd, {"N17"}, {"cardiometabolic": {"I50", "E11"}}, gap_days=30, min_patients=1)
    assert out.height == 1
    r = out.row(0, named=True)
    assert (r["target"], r["a"], r["bridge"], r["core_code"], r["n_patients"]) == ("cardiometabolic", "X", "N17", "I50", 1)


def test_bridge_chains_min_patients_filter():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1, 1], dtype=pl.Int64), "category": ["X", "N17", "I50"],
                       "first_date": [date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1)]})
    assert motifs.bridge_chains(fd, {"N17"}, {"t": {"I50"}}, 30, min_patients=2).height == 0


def test_rewire_preserves_in_and_out_degrees(synth):
    dg = nx.DiGraph()
    dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:60])
    h = motifs._rewire(dg, 0)
    assert dict(h.in_degree()) == dict(dg.in_degree()) and dict(h.out_degree()) == dict(dg.out_degree())
    assert set(h.edges()) != set(dg.edges())
    assert not any(u == v for u, v in h.edges())


def test_rewire_keeps_null_free_of_reciprocal_edges_when_input_is(synth):
    dg = nx.DiGraph()
    dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:60])   # one direction per pair
    assert not any(dg.has_edge(v, u) for u, v in dg.edges())
    h = motifs._rewire(dg, 3)
    assert not any(h.has_edge(v, u) for u, v in h.edges())


def _dg_from_synth(synth, k=60):
    dg = nx.DiGraph()
    dg.add_edges_from([(u, v) for u, v in synth.graph.edges()][:k])
    return dg


def test_rewire_restricted_to_allowed_pairs(synth):
    dg = _dg_from_synth(synth)
    allowed = {frozenset(e) for e in synth.graph.edges()}
    h = motifs._rewire(dg, 0, allowed_pairs=allowed)
    assert all(frozenset((u, v)) in allowed for u, v in h.edges())
    assert dict(h.in_degree()) == dict(dg.in_degree()) and dict(h.out_degree()) == dict(dg.out_degree())


def test_null_census_reports_both_nulls_and_hides_disconnected_z(synth):
    dg = _dg_from_synth(synth)
    allowed = {frozenset(e) for e in synth.graph.edges()}
    out = motifs.null_census(dg, n=4, seed=0, allowed_pairs=allowed)
    assert {"triad", "observed", "null_mean", "null_sd", "z", "p_emp", "null_u_mean", "null_u_sd", "z_u", "p_emp_u"} <= set(out.columns)
    disc = out.filter(pl.col("triad").is_in(["003", "012", "102"]))
    assert disc["z"].is_null().all() and disc["z_u"].is_null().all()
    assert out.filter(pl.col("triad") == "030T")["p_emp"].is_not_null().all()


def test_feed_forward_middle_into_core_columns():
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"),     # chain into core code C, A not in core
                      ("K", "B"), ("B", "C2"), ("K", "C2")])  # chain whose start K is a core member
    cores = {"t": {"C", "C2", "K"}}
    out = motifs.feed_forward_middle_share(g, {"B"}, cores)
    r = out.row(0, named=True)
    assert r["n_middle"] == 2 and r["n_middle_into_core_t"] == 1


def test_bridge_chains_entry_variant_and_graph_flag():
    fd_full = pl.DataFrame({"person_id": pl.Series([1, 1, 1, 2, 2, 2, 2], dtype=pl.Int64),
                            "category": ["X", "N17", "I50", "X", "N17", "I50", "E11"],
                            "first_date": [date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1),
                                           date(2018, 1, 1), date(2018, 3, 1), date(2018, 6, 1), date(2017, 6, 1)]})  # patient 2 already in core
    out = motifs.bridge_chains(fd_full, {"N17"}, {"cardiometabolic": {"I50", "E11"}}, 30, 1, fd_full=fd_full, node_set={"N17", "I50"})
    r = out.row(0, named=True)
    assert (r["n_patients"], r["n_patients_entry"], r["a_in_graph"]) == (2, 1, False)


def test_bridge_chains_gap_zero_is_strict():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1, 1], dtype=pl.Int64), "category": ["X", "N17", "I50"],
                       "first_date": [date(2018, 1, 1), date(2018, 1, 1), date(2018, 6, 1)]})
    assert motifs.bridge_chains(fd, {"N17"}, {"t": {"I50"}}, 0, 1).height == 0
