from datetime import date

import polars as pl

from bridgegnn import cohort


def _rows():
    return pl.DataFrame({
        "person_id": [1, 1, 2, 3, 4, 5], "index_date": [date(2018, 1, 1)] * 6,
        "target": ["cardiometabolic"] * 6, "split": ["train", "train", "val", "test", "test", "test"],
        "positive": [True, False, False, True, False, True], "age_at_index": [70.0, 70.0, 40.0, 66.0, 30.0, 50.0],
        "sex": ["female", "female", "male", "female", "male", "female"], "days_to_event": [100, None, None, 30, None, 200],
    })


def test_cohort_table_counts_rows_patients_positives_and_test_shares():
    t = cohort.cohort_table(_rows())
    r = t.filter(pl.col("target") == "cardiometabolic").row(0, named=True)
    assert (r["n_rows_dev"], r["n_patients_dev"], r["n_pos_dev"]) == (3, 2, 1)
    assert (r["n_rows_test"], r["n_patients_test"], r["n_pos_test"]) == (3, 3, 2)
    assert abs(r["test_female_share"] - 2 / 3) < 1e-9 and abs(r["test_age65_share"] - 1 / 3) < 1e-9


def test_triggering_codes_take_the_first_core_code_inside_the_horizon():
    rows = _rows().filter(pl.col("positive"))
    first = pl.DataFrame({"person_id": [1, 1, 3, 5, 5], "category": ["E11", "I10", "I50", "E11", "N18"],
                          "first_date": [date(2018, 4, 11), date(2018, 2, 1), date(2018, 1, 31), date(2018, 7, 20), date(2019, 6, 1)]})
    core = {"cardiometabolic": ["E11", "I10", "I50"]}
    t = cohort.triggering_codes(rows, first, core, horizon_days=365)
    got = dict(zip(t["category"].to_list(), t["n"].to_list()))
    assert got == {"I10": 1, "I50": 1, "E11": 1}    # person 1: I10 (Feb) before E11 (Apr); N18 is not core
    assert t["share"].sum() == 1.0


def test_subgraph_sizes_use_masked_graph_codes_in_the_window():
    rc = pl.DataFrame({"person_id": [1, 1, 1, 2], "index_date": [date(2018, 1, 1)] * 4, "category": ["A", "B", "C", "D"],
                       "in_lb1095": [True, True, False, True], "in_main_graph": [True, True, True, False],
                       "masked_cardiometabolic": [False, True, False, False]})
    rows = _rows().filter(pl.col("person_id").is_in([1, 2])).unique(subset=["person_id"], keep="first")
    t = cohort.subgraph_sizes(rows, rc, ["cardiometabolic"])
    r = t.row(0, named=True)
    assert r["median_nodes"] == 0.5 and r["share_empty"] == 0.5    # person 1 keeps only A; person 2 keeps nothing


def test_bridge_carriers_count_rows_with_each_bridge_code_in_the_window():
    rc = pl.DataFrame({"person_id": [1, 1, 2, 3], "index_date": [date(2018, 1, 1)] * 4, "category": ["N17", "A41", "N17", "E87"],
                       "in_lb1095": [True, False, True, True], "in_main_graph": [True] * 4,
                       "masked_cardiometabolic": [False] * 4})
    rows = _rows().unique(subset=["person_id"], keep="first")   # patients 1-5, target cardiometabolic
    t = cohort.bridge_carriers(rows, rc, {"cardiometabolic": ["N17", "A41", "E87"]})
    got = {r["category"]: (r["n_rows"], r["n_pos_rows"]) for r in t.iter_rows(named=True)}
    assert got == {"N17": (2, 1), "A41": (0, 0), "E87": (1, 1)}   # A41 only outside the window; patients 1 and 3 are positives
