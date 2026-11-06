from datetime import date, timedelta

import polars as pl
import pytest

from bridgegnn import contract, inputs, rows

TASK_CFG = {"task": {"definition": "core", "horizon_months": 12, "exclude_censored": True},
            "windows": {"lookback_days": [365, 1095, 0], "primary_lookback": 1095, "era_start": "2015-10-01",
                        "recent_days": 90, "visit_window_days": 365},
            "masking": {"input_exclusion": ["core", "chapter_block"]}}


def nodes_with(cats):
    return pl.DataFrame({"category": cats, "name": cats, "chapter": ["x"] * len(cats)})


@pytest.fixture(scope="module")
def built(synth):
    labels = inputs.task_labels(synth.labels, TASK_CFG)
    idx = labels.select("person_id", "index_date").unique()
    masks = rows.mask_sets(synth.target_defs, TASK_CFG)
    rc = rows.build_row_codes(idx, synth.codes_long, synth.nodes, masks, TASK_CFG)
    r = rows.build_rows(labels, rc, synth.visits, synth.demo, synth.codes_long, TASK_CFG)
    return labels, idx, masks, rc, r


def test_row_codes_matches_contract(built):
    _, _, _, rc, _ = built
    contract.validate(rc, "row_codes")
    assert rc.height > 0


def test_no_code_after_index(built):
    _, _, _, rc, _ = built
    assert (rc["days_since_last"] >= 0).all()
    assert (rc["days_since_first"] >= rc["days_since_last"]).all()


def test_code_on_index_day_is_included():
    one = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "category": ["I10"], "condition_date": [date(2019, 1, 1)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [date(2019, 1, 1)]})
    rc = rows.build_row_codes(idx, one, nodes_with(["I10"]), {}, TASK_CFG)
    assert rc.height == 1 and rc["days_since_last"][0] == 0 and rc["in_lb365"][0] and rc["in_main_graph"][0]


def test_code_after_index_is_excluded():
    one = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "category": ["I10"], "condition_date": [date(2019, 1, 2)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [date(2019, 1, 1)]})
    assert rows.build_row_codes(idx, one, nodes_with(["I10"]), {}, TASK_CFG).height == 0


def test_non_graph_category_kept_with_flag():
    one = pl.DataFrame({"person_id": pl.Series([1, 1], dtype=pl.Int64), "category": ["I10", "Z79"],
                        "condition_date": [date(2018, 1, 1), date(2018, 6, 1)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [date(2019, 1, 1)]})
    rc = rows.build_row_codes(idx, one, nodes_with(["I10"]), {}, TASK_CFG).sort("category")
    assert rc["category"].to_list() == ["I10", "Z79"]
    assert rc["in_main_graph"].to_list() == [True, False]


def test_mask_sets_union_core_and_block(synth):
    m = rows.mask_sets(synth.target_defs, TASK_CFG)
    td = synth.target_defs.filter(pl.col("target") == "cardiometabolic")
    assert m["cardiometabolic"] == set(td.filter(pl.col("in_core") | pl.col("in_chapter_block"))["category"])
    core_only = rows.mask_sets(synth.target_defs, {"masking": {"input_exclusion": ["core"]}})
    assert core_only["cardiometabolic"] == set(td.filter(pl.col("in_core"))["category"])


def test_mask_columns_follow_mask_sets(built):
    _, _, masks, rc, _ = built
    for t, s in masks.items():
        assert set(rc.filter(pl.col(f"masked_{t}"))["category"]) <= s
        assert rc.filter(pl.col("category").is_in(list(s)) & ~pl.col(f"masked_{t}")).height == 0


def test_row_codes_shared_across_targets(built):
    _, _, _, rc, _ = built
    assert rc.select("person_id", "index_date", "category").is_unique().all()


def test_rows_matches_contract_and_counts(built):
    labels, _, _, _, r = built
    contract.validate(r, "rows")
    assert r.height == labels.height
    assert (r["n_codes_lb365"] <= r["n_codes_lb1095"]).all() and (r["n_codes_lb1095"] <= r["n_codes_lball"]).all()
    assert (r["age_at_index"] > 0).all() and (r["n_visit_days_365"] >= 1).all()
    assert set(r["sex"]) <= {"female", "male"}


def test_row_with_only_masked_codes_survives():
    one = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "category": ["E11"], "condition_date": [date(2018, 1, 1)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [date(2019, 1, 1)]})
    masks = {"cardiometabolic": {"E11"}}
    rc = rows.build_row_codes(idx, one, nodes_with(["E11"]), masks, TASK_CFG)
    labels = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "split": ["train"], "index_date": [date(2019, 1, 1)],
                           "target": ["cardiometabolic"], "positive": [False], "days_to_event": pl.Series([None], dtype=pl.Int64),
                           "history_days": pl.Series([800], dtype=pl.Int64), "followup_days": pl.Series([400], dtype=pl.Int64)})
    visits = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "visit_date": [date(2018, 12, 1)]})
    demo = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "sex": ["female"], "birth_date": [date(1960, 1, 1)]})
    r = rows.build_rows(labels, rc, visits, demo, one, TASK_CFG)
    assert r.height == 1 and rc["masked_cardiometabolic"][0]
    assert r["n_codes_lb1095"][0] == 0 and r["n_codes_lball"][0] == 0 and r["n_rows_lb1095"][0] == 0


def test_lookback_window_boundary():
    idx_d = date(2019, 1, 1)
    one = pl.DataFrame({"person_id": pl.Series([1, 1], dtype=pl.Int64), "category": ["A01", "A02"],
                        "condition_date": [idx_d - timedelta(days=365), idx_d - timedelta(days=364)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [idx_d]})
    rc = rows.build_row_codes(idx, one, nodes_with(["A01", "A02"]), {}, TASK_CFG).sort("category")
    assert rc["in_lb365"].to_list() == [False, True]


def test_mask_sets_includes_in_range_category_that_is_not_a_node(synth):
    blocks = {"cardiometabolic": [["I20", "I25"]], "cancer_metastatic": [], "cancer_hematologic": [], "mental_health": []}
    m = rows.mask_sets(synth.target_defs, TASK_CFG, chapter_blocks=blocks, categories=["I22", "I30", "Z79"])
    assert "I22" in m["cardiometabolic"] and "I30" not in m["cardiometabolic"] and "Z79" not in m["cardiometabolic"]
    core = set(synth.target_defs.filter((pl.col("target") == "cardiometabolic") & pl.col("in_core"))["category"])
    assert core <= m["cardiometabolic"]


def test_row_codes_record_counts_per_window():
    d = date(2019, 1, 1)
    one = pl.DataFrame({"person_id": pl.Series([1] * 4, dtype=pl.Int64), "category": ["I10"] * 4,
                        "condition_date": [d, d - timedelta(days=100), d - timedelta(days=400), d - timedelta(days=1100)]})
    idx = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64), "index_date": [d]})
    rc = rows.build_row_codes(idx, one, nodes_with(["I10"]), {}, TASK_CFG)
    assert rc["n_rec_lb365"][0] == 2 and rc["n_rec_lb1095"][0] == 3 and rc["n_rec_all"][0] == 4


def test_rows_counts_exclude_masked_codes_of_own_target(built):
    _, _, _, rc, r = built
    one = r.filter(pl.col("target") == "cardiometabolic").row(0, named=True)
    sub = rc.filter((pl.col("person_id") == one["person_id"]) & (pl.col("index_date") == one["index_date"]))
    expected = sub.filter(pl.col("in_lb1095") & pl.col("in_main_graph") & ~pl.col("masked_cardiometabolic")).height
    assert one["n_codes_lb1095"] == expected
    assert one["n_rows_lb1095"] == sub.filter(~pl.col("masked_cardiometabolic"))["n_rec_lb1095"].sum()
