"""Run the configured GNN ablations and tabular baselines (step 07)."""
from __future__ import annotations

import argparse

import yaml

from bridgegnn import config, contract, driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="run just these experiment IDs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-baselines", action="store_true")
    args = parser.parse_args()
    cfg = config.load()
    grid_path = config.REPO_ROOT / "config" / "experiments.yaml"
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    try:
        jobs = driver.expand(grid, cfg, args.ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only_baselines:
        jobs = [j for j in jobs if j["script"] == driver.SCRIPTS["baseline"]]
    if args.ids:
        skipped = set(args.ids) - {j["run_id"] for j in jobs}
        if skipped:
            print(f"Skipped by driver: {', '.join(sorted(skipped))}")
    out = config.REPO_ROOT / cfg["paths"]["outputs"]
    if not args.dry_run and any(job["subsample_file"] for job in jobs):
        rows = contract.read(out / "rows.parquet", "rows")
        driver.write_subsamples(jobs, rows, cfg)
    codes = driver.run(jobs, args.dry_run, out / "logs")
    failures = sum(code != 0 for code in codes)
    print(f"{len(codes)} jobs, {failures} failed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
