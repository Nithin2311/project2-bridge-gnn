from datetime import date

import polars as pl

from bridgegnn import contract, directed


def fd_from_synth(synth, drop_day_one=True):
    pc = synth.codes_long.group_by("person_id", "category").agg(pl.col("condition_date").min().alias("first_date"))
    return directed.first_dates(pc, synth.split, ["train", "val"], drop_day_one)


def test_first_dates_excludes_test_patients(synth):
    fd = fd_from_synth(synth)
    test_ids = set(synth.split.filter(pl.col("split") == "test")["person_id"])
    assert not (set(fd["person_id"]) & test_ids)
    assert fd.columns == ["person_id", "category", "first_date"]


def test_first_dates_drops_day_one_codes(synth):
    fd_all = fd_from_synth(synth, drop_day_one=False)
    fd = fd_from_synth(synth, True)
    first_day = fd_all.group_by("person_id").agg(pl.col("first_date").min().alias("d0"))
    j = fd.join(first_day, on="person_id")
    assert (j["first_date"] > j["d0"]).all() and fd.height < fd_all.height


def test_precedence_counts_planted_pair(synth):
    a, b = synth.precedence_pair
    c = directed.precedence_counts(fd_from_synth(synth, False), [(a, b)], gap_days=30).row(0, named=True)
    assert c["n_ab"] > c["n_ba"] and c["n_ab"] + c["n_ba"] + c["n_tie"] >= 10


def test_precedence_gap_turns_close_pairs_into_ties():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1, 2, 2], dtype=pl.Int64), "category": ["A", "B", "A", "B"],
                       "first_date": [date(2018, 1, 1), date(2018, 1, 10), date(2018, 1, 1), date(2018, 3, 1)]})
    c0 = directed.precedence_counts(fd, [("A", "B")], 0).row(0, named=True)
    c30 = directed.precedence_counts(fd, [("A", "B")], 30).row(0, named=True)
    assert (c0["n_ab"], c0["n_tie"]) == (2, 0) and (c30["n_ab"], c30["n_tie"]) == (1, 1)


def test_precedence_counts_pair_order_is_canonical():
    fd = pl.DataFrame({"person_id": pl.Series([1, 1], dtype=pl.Int64), "category": ["A", "B"],
                       "first_date": [date(2018, 1, 1), date(2018, 6, 1)]})
    c = directed.precedence_counts(fd, [("B", "A")], 0).row(0, named=True)
    assert (c["a"], c["b"], c["n_ab"], c["n_ba"]) == ("B", "A", 0, 1)


def test_directed_edges_all_ties_yield_nothing():
    counts = pl.DataFrame({"a": ["A"], "b": ["B"], "n_ab": [0], "n_ba": [0], "n_tie": [40]})
    assert directed.directed_edges(counts, min_total=20, alpha=0.05).height == 0


def test_directed_edges_direction_and_bh():
    counts = pl.DataFrame({"a": ["A", "C", "E"], "b": ["B", "D", "F"],
                           "n_ab": [5, 30, 15], "n_ba": [40, 28, 14], "n_tie": [0, 0, 0]})
    out = directed.directed_edges(counts, 20, 0.05)
    assert out.height == 1
    r = out.row(0, named=True)
    assert (r["src"], r["dst"], r["n_fwd"], r["n_rev"]) == ("B", "A", 40, 5)
    contract.validate(out.with_columns(pl.lit(0.3).alias("phi")), "edges_directed")


def test_day_one_sensitivity_counts_more(synth):
    a, b = synth.precedence_pair
    n_drop = directed.precedence_counts(fd_from_synth(synth, True), [(a, b)], 30).row(0, named=True)
    n_keep = directed.precedence_counts(fd_from_synth(synth, False), [(a, b)], 30).row(0, named=True)
    assert n_keep["n_ab"] + n_keep["n_ba"] >= n_drop["n_ab"] + n_drop["n_ba"]


def test_to_digraph_and_tables(synth):
    g_u = synth.graph
    pairs = list(g_u.edges())
    counts = directed.precedence_counts(fd_from_synth(synth, False), pairs, 30)
    de = directed.attach_phi(directed.directed_edges(counts, 5, 0.5), g_u)
    assert de.height > 0
    contract.validate(de, "edges_directed")
    dg = directed.to_digraph(de, g_u)
    assert dg.number_of_edges() == de.height
    assert all({"phi", "weight", "distance", "n_fwd"} <= set(d) for _, _, d in dg.edges(data=True))
    assert dg.nodes["%s" % de["src"][0]].get("name")
    cores = synth.targets["core"]
    into = directed.edges_into_cores(de, cores, top_k=5)
    assert {"target", "src", "dst", "n_fwd", "share", "q_bh"} <= set(into.columns)
    assert all(r["dst"] in cores[r["target"]] and r["src"] not in cores[r["target"]] for r in into.iter_rows(named=True))
    thr = directed.edges_through_bridges(de, {de["src"][0]})
    assert set(thr["role"]) <= {"into_bridge", "out_of_bridge"} and thr.height >= 1
    s = directed.summary(de, g_u)
    assert {"n_directed", "n_undirected", "share_directed", "median_share", "n_tested"} <= set(s)
