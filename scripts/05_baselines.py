"""Fit and score one baseline and target on the canonical patient split (step 05)."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

import polars as pl
import yaml

from bridgegnn import config, contract, tabular

LOGGER = logging.getLogger(__name__)


def upsert_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate CLI invocations can finish at the same time. Serialize the CSV read/replace.
    with path.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        row = pl.DataFrame([result], schema=contract.SCHEMAS["results"])
        if path.exists():
            row = pl.concat([contract.read(path, "results"), row]).unique(
                ["run_id", "target", "seed"], keep="last", maintain_order=True)
        pending = path.with_suffix(".pending.csv")
        contract.write(row, pending, "results")
        pending.replace(path)


def run(args, cfg: dict, out: Path) -> None:
    features = out / "features"
    if args.synthetic:
        from bridgegnn.features import build_synthetic

        build_synthetic(cfg, features, verbose=False)
    tables = {split: contract.read(features / f"tabular_{args.target}_{split}.parquet", "tabular")
              for split in (*cfg["split"]["dev"], *cfg["split"]["test"])}
    groups = json.loads((features / "feature_groups.json").read_text(encoding="utf-8"))
    experiments = yaml.safe_load((config.REPO_ROOT / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    family = next((e["family"] for e in experiments["experiments"] if e["id"] == args.run_id), "baseline")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs / f"05_{args.target}_{args.model}.log", level=logging.INFO, encoding="utf-8")
    path = out / "results" / "baselines.csv"
    cached = None
    n_dev = sum(tables[split].height for split in cfg["split"]["dev"])
    reusable = not args.subsample_file and tabular.seed_invariant(args.model, n_dev)
    for seed in args.seeds:
        subset = None
        if args.subsample_file:
            subset = set(map(int, Path(args.subsample_file.format(seed=seed)).read_text().split()))
        if cached is None:
            preds, result = tabular.run(args.model, args.target, tables, cfg, seed, run_id=args.run_id,
                                        family=family, subset=subset, feature_groups=groups)
            if reusable:
                cached = (preds, result)
        else:
            preds, original = cached
            preds = preds.with_columns(pl.lit(seed, dtype=pl.Int32).alias("seed"))
            result = {**original, "seed": seed}
            LOGGER.info("seed %s reuses deterministic fit from seed %s", seed, original["seed"])
        contract.write(preds, out / "predictions" / f"{args.run_id}_{args.target}_seed{seed}.parquet",
                       "predictions")
        upsert_result(path, result)
        LOGGER.info("%s", json.dumps(result, allow_nan=False))
        print(json.dumps(result, allow_nan=False), flush=True)


def main() -> None:
    cfg = config.load()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="defaults to B-<model>")
    parser.add_argument("--target", required=True, choices=contract.TARGETS)
    parser.add_argument("--model", required=True, choices=tabular.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--subsample-file", help="path template with {seed}; one development patient per line")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--outputs", type=Path, help="override the Project 2 output directory")
    args = parser.parse_args()
    args.run_id = args.run_id or f"B-{args.model}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id must contain only letters, numbers, hyphens or underscores")
    if args.synthetic and not args.outputs:
        with tempfile.TemporaryDirectory(prefix="bridgegnn-baselines-") as tmp:
            try:
                run(args, cfg, Path(tmp))
            finally:
                logging.shutdown()
    else:
        run(args, cfg, args.outputs or config.outputs_dir(cfg))


if __name__ == "__main__":
    main()
