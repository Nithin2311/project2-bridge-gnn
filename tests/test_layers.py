import polars as pl

from bridgegnn import layers


def test_bh_keeps_edges_passing_all_floors():
    edges_all = pl.DataFrame({"u": ["A", "A", "B", "C", "A"], "v": ["B", "C", "C", "D", "Z"],
                              "n_both": [50, 30, 12, 9, 50], "odds_ratio": [3.0, 1.5, 1.1, 2.0, 3.0],
                              "p_fisher": [1e-10, 1e-3, 1e-4, 1e-6, 1e-10], "phi": [.4, .2, .1, .3, .4]})
    out, family_n = layers.bh_edges(edges_all, {"A", "B", "C", "D"}, {"layers": {"bh_alpha": 0.05}, "p1_edges": {"min_odds_ratio": 1.2, "min_shared_patients": 10}})
    assert family_n == 5
    # B-C fails the OR floor, C-D fails the n_both floor, A-Z has an endpoint outside the node set
    assert set(zip(out["u"], out["v"])) == {("A", "B"), ("A", "C")}
    assert "q_bh" in out.columns and (out["q_bh"] < 0.05).all()


def test_graph_from_edges_uses_template_attributes(synth):
    g = synth.graph
    edges = pl.DataFrame({"u": [u for u, _ in g.edges()][:5], "v": [v for _, v in g.edges()][:5], "phi": [0.1, 0.2, 0.3, 0.4, 0.5]})
    h = layers.graph_from_edges(edges, g)
    assert h.number_of_edges() == 5 and set(h.nodes) == set(g.nodes)
    u, v = edges["u"][0], edges["v"][0]
    assert h[u][v]["weight"] == 0.1 and abs(h[u][v]["distance"] - 10.0) < 1e-9 and h.nodes[u]["name"]


def test_shuffled_graph_same_size_no_loops_degrees_differ(synth):
    g = synth.graph
    h = layers.shuffled_graph(g, 0)
    assert set(h.nodes) == set(g.nodes) and h.number_of_edges() == g.number_of_edges()
    assert not any(u == v for u, v in h.edges())
    assert sorted(d for _, d in h.degree()) != sorted(d for _, d in g.degree())
    assert sorted(d["weight"] for *_, d in h.edges(data=True)) == sorted(d["weight"] for *_, d in g.edges(data=True))


def test_rewired_graphs_preserve_degrees(synth):
    g = synth.graph
    hs = layers.rewired_graphs(g, 2, 0)
    assert len(hs) == 2
    assert all(sorted(dict(h.degree()).items()) == sorted(dict(g.degree()).items()) for h in hs)
    assert any(set(h.edges()) != set(g.edges()) for h in hs)


def test_layer_stats_keys(synth):
    s = layers.layer_stats(synth.graph)
    assert set(s) == {"n_nodes", "n_edges", "density", "n_isolates", "mean_degree"}
    assert s["n_edges"] == synth.graph.number_of_edges()


def test_bh_family_is_all_tested_pairs_then_floors():
    # the row failing the floors still belongs to the BH family (P1's n_tests = all pairs)
    edges_all = pl.DataFrame({"u": ["A", "A", "B"], "v": ["B", "C", "C"], "n_both": [50, 30, 5],
                              "odds_ratio": [3.0, 1.5, 1.1], "p_fisher": [1e-10, 0.03, 0.9], "phi": [.4, .2, .1]})
    cfg = {"layers": {"bh_alpha": 0.05}, "p1_edges": {"min_odds_ratio": 1.2, "min_shared_patients": 10}}
    out, family_n = layers.bh_edges(edges_all, {"A", "B", "C"}, cfg)
    assert family_n == 3
    # BH over 3 pairs: p=0.03 has rank 2 -> q = 0.03*3/2 = 0.045 < 0.05 keeps A-C; floor-first BH would give the same here,
    # so also check the q value itself reflects the 3-pair family
    assert abs(out.filter(pl.col("v") == "C")["q_bh"][0] - 0.045) < 1e-9
