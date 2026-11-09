from datetime import date

import polars as pl

from bridgegnn.gnn import data


def _rows():
    return pl.DataFrame({"person_id": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
                         "index_date": [date(2018, 4, 1), date(2019, 12, 31), date(2020, 1, 1), date(2020, 6, 30), date(2020, 7, 1), date(2021, 3, 1)],
                         "split": ["train", "val", "train", "test", "test", "test"]})


def test_index_window_filters_dev_and_test_rows():
    out = data.filter_rows(_rows(), index_from="2020-01-01", index_to=None, test_index_from=None)
    assert out["person_id"].to_list() == [3, 4, 5, 6]
    out = data.filter_rows(_rows(), index_from=None, index_to="2019-12-31", test_index_from=None)
    assert out["person_id"].to_list() == [1, 2]


def test_backtest_windows_leave_no_early_test_row():
    out = data.filter_rows(_rows(), index_from=None, index_to="2019-12-31", test_index_from="2020-07-01")
    dev = out.filter(pl.col("split") != "test"); test = out.filter(pl.col("split") == "test")
    assert dev["person_id"].to_list() == [1, 2]
    assert test["person_id"].to_list() == [5, 6] and (test["index_date"] >= date(2020, 7, 1)).all()


def test_no_filters_is_identity():
    assert data.filter_rows(_rows(), None, None, None).height == 6
