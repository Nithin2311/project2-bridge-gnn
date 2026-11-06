import polars as pl

from bridgegnn import db


def test_connect_registers_views_from_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame({"person_id": [1, 2], "gender_concept_id": [8532, 8507]}).write_csv(data / "person.csv")
    cfg = {"paths": {"data_env": "EHRSHOT_DATA_TEST", "data_default": str(data), "cache": str(tmp_path / "c.duckdb")}}
    monkeypatch.delenv("EHRSHOT_DATA_TEST", raising=False)
    con = db.connect(cfg)
    assert con.execute("select count(*) from person").fetchone()[0] == 2


def test_connect_refreshes_stale_view(tmp_path):
    data1, data2 = tmp_path / "d1", tmp_path / "d2"
    data1.mkdir(); data2.mkdir()
    pl.DataFrame({"person_id": [1]}).write_csv(data1 / "person.csv")
    pl.DataFrame({"person_id": [1, 2, 3]}).write_csv(data2 / "person.csv")
    cache = str(tmp_path / "c.duckdb")
    db.connect({"paths": {"data_env": "X_UNSET", "data_default": str(data1), "cache": cache}}).close()
    con = db.connect({"paths": {"data_env": "X_UNSET", "data_default": str(data2), "cache": cache}})
    assert con.execute("select count(*) from person").fetchone()[0] == 3


def test_register_category_map_builds_cat_map():
    import duckdb
    con = duckdb.connect()
    con.execute("create table condition_occurrence(person_id bigint, condition_source_value varchar, condition_start_DATE date)")
    con.execute("insert into condition_occurrence values (1,'I10.9',date '2016-01-01'), (1,'E11.65',date '2016-01-01'), (1,NULL,date '2016-01-01'), (2,'I10.9',date '2017-01-01')")
    db.register_category_map(con)
    rows = sorted(con.sql("select sv, category from cat_map").fetchall())
    assert ("I10.9", "I10") in rows and ("E11.65", "E11") in rows
    assert len(rows) == 2
