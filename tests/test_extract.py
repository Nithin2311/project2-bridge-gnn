from datetime import date

import duckdb
import pytest

from bridgegnn import extract


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""
        create table condition_occurrence(person_id bigint, condition_source_value varchar, condition_start_DATE date);
        insert into condition_occurrence values
          (1, 'I10.9',  date '2016-01-01'),
          (1, 'I10.9',  date '2016-01-01'),      -- exact duplicate
          (1, 'I10.0',  date '2016-01-01'),      -- same category, same date -> one row
          (1, 'E11.9',  date '2014-01-01'),      -- pre-era
          (1, 'Z79.4',  date '2018-05-05'),
          (2, 'I10.9',  date '2019-01-01'),
          (2, 'N17.9',  date '2021-01-01'),      -- after death
          (3, 'XYZ',    date '2019-01-01');      -- unmappable
        create table death(person_id bigint, death_DATE date);
        insert into death values (2, date '2020-06-30');
        create table visit_occurrence(person_id bigint, visit_start_DATE date, visit_concept_id bigint);
        insert into visit_occurrence values (1, date '2016-01-01', 9202), (1, date '2016-01-01', 9201),
                                            (2, date '2019-01-01', 9202), (2, date '2021-01-01', 9202);
        create table person(person_id bigint, gender_concept_id bigint, birth_DATETIME timestamp);
        insert into person values (1, 8532, timestamp '1960-05-01 00:00:00'), (2, 8507, timestamp '1950-01-01 00:00:00'),
                                  (3, 8507, timestamp '1970-01-01 00:00:00');
        create temp table cat_map(sv varchar, category varchar);
        insert into cat_map values ('I10.9','I10'), ('I10.0','I10'), ('E11.9','E11'), ('Z79.4','Z79'), ('N17.9','N17');
    """)
    return c


def test_codes_long_drops_pre_era_post_death_and_duplicates(con):
    out = extract.codes_long(con, [1, 2, 3], "2015-10-01").sort("person_id", "category", "condition_date")
    assert out.columns == ["person_id", "category", "condition_date"]
    assert out.rows() == [(1, "I10", date(2016, 1, 1)), (1, "Z79", date(2018, 5, 5)), (2, "I10", date(2019, 1, 1))]


def test_codes_long_respects_person_filter(con):
    assert set(extract.codes_long(con, [2], "2015-10-01")["person_id"]) == {2}


def test_visits_distinct_dates_after_death_dropped(con):
    out = extract.visits(con, [1, 2]).sort("person_id", "visit_date")
    assert out.columns == ["person_id", "visit_date"]
    assert out.rows() == [(1, date(2016, 1, 1)), (2, date(2019, 1, 1))]


def test_demo_maps_sex_and_birth(con):
    out = extract.demo(con, [1, 2]).sort("person_id")
    assert out.columns == ["person_id", "sex", "birth_date"]
    assert out.rows() == [(1, "female", date(1960, 5, 1)), (2, "male", date(1950, 1, 1))]
