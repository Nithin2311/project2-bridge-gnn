from bridgegnn import config


def test_load_has_required_sections():
    cfg = config.load()
    assert cfg["task"]["horizon_months"] == 12
    assert cfg["windows"]["primary_lookback"] == 1095
    assert set(cfg["task"]["primary_targets"]) == {"cardiometabolic", "cancer_hematologic", "mental_health"}


def test_p1_outputs_path_is_absolute():
    assert config.p1_outputs(config.load()).is_absolute()
