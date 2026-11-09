import numpy as np
import polars as pl
import pytest

from bridgegnn import config as _config
from bridgegnn import nodefeat as _nodefeat
from bridgegnn import synth as _synth


@pytest.fixture(scope="session")
def synth():
    return _synth.make(0)


@pytest.fixture(scope="session")
def cfg():
    return _config.load()


@pytest.fixture(scope="session")
def node_feats(synth):
    rng = np.random.default_rng(1)
    n = synth.nodes.height
    cats = synth.nodes["category"].to_list()
    df = synth.nodes.select("category", "name", "chapter")
    for c in _nodefeat.STRUCT_COLS:
        df = df.with_columns(pl.Series(c, rng.normal(size=n)))
    df = df.with_columns(pl.Series("community_seed0", rng.choice([0, 1, 2, 7], size=n).tolist()),
                         pl.Series("bridge_survivor", [c in cats[:2] for c in cats]))
    # a null in one STRUCT column, as in the real file (ccir_chronic_share is null for unmapped categories)
    vals = df["ccir_chronic_share"].to_list(); vals[0] = None
    return df.with_columns(pl.Series("ccir_chronic_share", vals, dtype=pl.Float64))


