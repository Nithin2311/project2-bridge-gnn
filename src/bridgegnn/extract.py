"""Step 01 SQL: the three per-patient frames the row builders consume, pulled from the EHRShot views.

Requires a connection with the EHRShot views (bridgegnn.db.connect) and the temp table cat_map
(bridgegnn.db.register_category_map).
"""
from __future__ import annotations

import polars as pl


def _ids(con, person_ids: list[int]) -> None:
    con.register("_pids_df", pl.DataFrame({"person_id": pl.Series(person_ids, dtype=pl.Int64)}).to_arrow())
    con.execute("create or replace temp table _pids as select person_id from _pids_df")


def codes_long(con, person_ids: list[int], era_start: str) -> pl.DataFrame:
    """Distinct (person_id, category, condition_date) in the ICD-10 era, post-death rows dropped, all categories."""
    _ids(con, person_ids)
    return con.sql(f"""
        select distinct co.person_id, m.category, co.condition_start_DATE as condition_date
        from condition_occurrence co
        join cat_map m on m.sv = co.condition_source_value
        join _pids p on p.person_id = co.person_id
        left join death d on d.person_id = co.person_id
        where co.condition_start_DATE >= date '{era_start}'
          and (d.death_DATE is null or co.condition_start_DATE <= d.death_DATE)
        order by 1, 2, 3
    """).pl().with_columns(pl.col("person_id").cast(pl.Int64), pl.col("category").cast(pl.Utf8), pl.col("condition_date").cast(pl.Date))


def visits(con, person_ids: list[int]) -> pl.DataFrame:
    """Distinct (person_id, visit_date) over every visit type, post-death rows dropped."""
    _ids(con, person_ids)
    return con.sql("""
        select distinct v.person_id, v.visit_start_DATE as visit_date
        from visit_occurrence v
        join _pids p on p.person_id = v.person_id
        left join death d on d.person_id = v.person_id
        where v.visit_start_DATE is not null and (d.death_DATE is null or v.visit_start_DATE <= d.death_DATE)
        order by 1, 2
    """).pl().with_columns(pl.col("person_id").cast(pl.Int64), pl.col("visit_date").cast(pl.Date))


def demo(con, person_ids: list[int]) -> pl.DataFrame:
    """(person_id, sex, birth_date); sex from gender_concept_id 8532 female / 8507 male."""
    _ids(con, person_ids)
    return con.sql("""
        select pe.person_id,
               case pe.gender_concept_id when 8532 then 'female' when 8507 then 'male' else 'other' end as sex,
               cast(pe.birth_DATETIME as date) as birth_date
        from person pe join _pids p on p.person_id = pe.person_id
        order by 1
    """).pl().with_columns(pl.col("person_id").cast(pl.Int64), pl.col("sex").cast(pl.Utf8), pl.col("birth_date").cast(pl.Date))
