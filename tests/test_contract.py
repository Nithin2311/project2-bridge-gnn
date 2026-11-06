import polars as pl
import pytest

from bridgegnn import contract


def test_validate_accepts_matching_frame():
    df = pl.DataFrame({c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["rows"].items()})
    assert contract.validate(df, "rows").height == 0


def test_validate_rejects_missing_column():
    df = pl.DataFrame({"person_id": pl.Series([1], dtype=pl.Int64)})
    with pytest.raises(contract.ContractError, match="index_date"):
        contract.validate(df, "rows")


def test_validate_rejects_wrong_dtype():
    sch = contract.SCHEMAS["predictions"]
    df = pl.DataFrame({c: (pl.Series(["0"], dtype=pl.Utf8) if c == "p" else pl.Series([0], dtype=pl.Int64)) for c in sch})
    with pytest.raises(contract.ContractError, match="p"):
        contract.validate(df, "predictions")


def test_rows_rejects_extra_column():
    base = {c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["rows"].items()}
    base["surprise"] = pl.Series([], dtype=pl.Float64)
    with pytest.raises(contract.ContractError, match="surprise"):
        contract.validate(pl.DataFrame(base), "rows")


def test_tabular_allows_extra_feature_columns():
    base = {c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["tabular"].items()}
    base["c_I10"] = pl.Series([], dtype=pl.Float64)
    assert contract.validate(pl.DataFrame(base), "tabular") is not None


def test_write_read_roundtrip(tmp_path):
    df = pl.DataFrame({c: pl.Series([], dtype=t) for c, t in contract.SCHEMAS["node_index"].items()})
    p = tmp_path / "node_index.csv"
    contract.write(df, p, "node_index")
    assert contract.read(p, "node_index").columns == list(contract.SCHEMAS["node_index"])
