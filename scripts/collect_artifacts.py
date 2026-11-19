"""Copy the aggregate outputs into outputs/artifacts/ (see bridgegnn.artifacts.PATTERNS). Never patient-level files."""
from __future__ import annotations

import argparse
from pathlib import Path

from bridgegnn import artifacts, config


def main() -> None:
    cfg = config.load()
    out_default = config.outputs_dir(cfg)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=out_default)
    ap.add_argument("--dest", type=Path, default=out_default / "artifacts")
    args = ap.parse_args()
    copied = artifacts.collect(args.out, args.dest)
    print(f"collected {len(copied)} aggregate files into {args.dest}")


if __name__ == "__main__":
    main()
