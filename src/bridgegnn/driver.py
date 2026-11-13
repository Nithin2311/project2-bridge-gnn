"""Expand the configured experiment grid and run each target as a separate job."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from . import config

SCRIPTS = {"baseline": "scripts/05_baselines.py", "gnn": "scripts/06_gnn.py"}


def expand(experiments: dict, cfg: dict, ids: list[str] | None) -> list[dict]:
    entries = experiments["experiments"]
    if ids is not None:
        unknown = set(ids) - {e["id"] for e in entries}
        if unknown:
            raise ValueError(f"unknown experiment ids: {', '.join(sorted(unknown))}")
    jobs = []
    for entry in entries:
        run_id, base = entry["id"], entry["base"]
        if base == "script" or (ids is not None and run_id not in ids):
            continue
        if base not in SCRIPTS:
            raise ValueError(f"{run_id}: unknown base {base!r}; expected one of {sorted(SCRIPTS)} or 'script'")
        targets = cfg["task"]["primary_targets"] if entry["targets"] == "primary" else entry["targets"]
        overrides = entry.get("overrides", {}).copy()
        seeds = overrides.pop("seeds", cfg["seeds"])
        fraction = float(overrides.pop("subsample", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError(f"{run_id}: subsample must be in (0, 1]")
        if base == "gnn":
            overrides = {"arch": experiments.get("default_arch", "sage"), **overrides}
        subsample = None
        if fraction < 1:
            subsample = str(Path(cfg["paths"]["outputs"]) / "subsamples" / f"{run_id}_seed{{seed}}.txt")
            overrides["subsample-file"] = subsample
        for target in targets:
            argv = ["--run-id", run_id, "--target", target, "--seeds", *map(str, seeds)]
            for flag, value in overrides.items():
                argv.extend([f"--{flag}", str(value)])
            jobs.append({"run_id": run_id, "target": target, "script": SCRIPTS[base], "argv": argv,
                         "subsample_file": subsample, "subsample": fraction, "seeds": seeds})
    return jobs


def subsample_patients(person_ids: pl.Series, fraction: float, seed: int) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("subsample fraction must be in (0, 1]")
    ids = np.unique(person_ids.to_numpy())
    if ids.size == 0:
        raise ValueError("cannot subsample an empty development cohort")
    n = max(1, int(ids.size * fraction))
    return np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False)).tolist()


def write_subsamples(jobs: list[dict], rows: pl.DataFrame, cfg: dict) -> None:
    people = rows.filter(pl.col("split").is_in(cfg["split"]["dev"]))["person_id"]
    written = set()
    for job in jobs:
        template = job["subsample_file"]
        if template is None:
            continue
        for seed in job["seeds"]:
            path = config.REPO_ROOT / template.format(seed=seed)
            if path in written or path.exists():
                continue   # never overwrite an archived patient list: earlier runs must stay reproducible
            path.parent.mkdir(parents=True, exist_ok=True)
            ids = subsample_patients(people, job["subsample"], seed)
            path.write_text("".join(f"{pid}\n" for pid in ids), encoding="utf-8")
            written.add(path)


def run(jobs: list[dict], dry_run: bool, log_dir: Path) -> list[int]:
    codes = []
    for job in jobs:
        command = [sys.executable, job["script"], *job["argv"]]
        if dry_run:
            print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command))
            codes.append(0)
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"07_{job['run_id']}_{job['target']}.log"
        with path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, check=False, cwd=config.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                code = result.returncode
            except OSError as exc:
                log.write(f"{exc}\n")
                code = 1
        print(f"{job['run_id']} {job['target']}: exit {code}", flush=True)
        codes.append(code)
    return codes
