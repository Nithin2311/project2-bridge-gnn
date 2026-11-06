import pytest

from bridgegnn import config as _config, synth as _synth


@pytest.fixture(scope="session")
def synth():
    return _synth.make(0)


@pytest.fixture(scope="session")
def cfg():
    return _config.load()
