"""Load config/p2.yaml and resolve paths (mirrors bridgedx.config)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "p2.yaml"


def load(path: Path | str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def p1_repo(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    return (REPO_ROOT / cfg["paths"]["p1_repo"]).resolve()


def p1_outputs(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    return (REPO_ROOT / cfg["paths"]["p1_outputs"]).resolve()


def p1_config(cfg: dict | None = None) -> dict:
    """The frozen Project 1 config (option_a.yaml), for the P1 helpers that need it."""
    from bridgedx import config as p1c

    return p1c.load(p1_repo(cfg) / "config" / "option_a.yaml")


def data_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    env = os.environ.get(cfg["paths"]["data_env"])
    p = Path(env) if env else (REPO_ROOT / cfg["paths"]["data_default"])
    return p.resolve()


def outputs_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    p = REPO_ROOT / cfg["paths"]["outputs"]
    p.mkdir(parents=True, exist_ok=True)
    return p
