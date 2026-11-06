import polars as pl

from bridgegnn import synth


def test_synth_is_deterministic():
    a, b = synth.make(0), synth.make(0)
    assert a.labels.equals(b.labels) and a.codes_long.equals(b.codes_long)


def test_synth_has_all_splits_and_targets():
    s = synth.make(0)
    assert set(s.split["split"]) == {"train", "val", "test"}
    assert set(s.labels["target"]) == {"cardiometabolic", "cancer_metastatic", "cancer_hematologic", "mental_health"}
    assert s.labels.filter(pl.col("censored")).height > 0
    assert set(s.labels["definition"]) == {"core", "community", "chapter"}
    assert set(s.labels["horizon_months"]) == {6, 12}


def test_synth_nodes_subset_of_categories():
    s = synth.make(0)
    assert set(s.nodes["category"]) < set(s.codes_long["category"])


def test_synth_frames_have_expected_columns():
    s = synth.make(0)
    assert s.codes_long.columns == ["person_id", "category", "condition_date"]
    assert s.visits.columns == ["person_id", "visit_date"]
    assert s.demo.columns == ["person_id", "sex", "birth_date"]
    assert s.split.columns == ["person_id", "split"]
    assert {"target", "category", "in_core", "in_community", "in_chapter_block"} <= set(s.target_defs.columns)
    assert {"pair", "category", "bridge_betweenness", "bridge_pass"} <= set(s.bridges.columns)
    assert s.graph.number_of_nodes() == s.nodes.height


def test_synth_planted_precedence():
    s = synth.make(0)
    a, b = s.precedence_pair
    first = s.codes_long.group_by("person_id", "category").agg(pl.col("condition_date").min().alias("d"))
    fa = first.filter(pl.col("category") == a).select("person_id", pl.col("d").alias("da"))
    fb = first.filter(pl.col("category") == b).select("person_id", pl.col("d").alias("db"))
    both = fa.join(fb, on="person_id")
    assert both.height >= 10
    assert (both["da"] < both["db"]).mean() >= 0.65


def test_synth_nodes_all_is_superset_like_nodes_csv():
    s = synth.make(0)
    assert set(s.nodes["category"]) < set(s.nodes_all["category"])
    assert set(s.nodes_all["category"]) == set(s.codes_long["category"])
