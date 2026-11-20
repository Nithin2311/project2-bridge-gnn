"""Compare two artifact trees: `uv run python scripts/compare_runs.py OLD_ARTIFACTS NEW_ARTIFACTS [--tol 0.02] [--md out.md]`.

Exit 1 when any comparison exceeds the tolerance or a file / column is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bridgegnn import compare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--tol", type=float, default=0.02, help="max AUROC/AUPRC drift tolerated for torch runs")
    ap.add_argument("--md", type=Path, help="write the comparison table (non-exact rows) as Markdown")
    ap.add_argument("--exclude-runs", nargs="*", default=[], help="run_id prefixes dropped from both trees before comparing")
    args = ap.parse_args()
    df = compare.compare_dirs(args.old, args.new, args.tol, args.exclude_runs)
    md = compare.to_markdown(df)
    print(md)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
    s = compare.summary(df)
    return 1 if (s["exceeds"] or s["missing_file"] or s["missing_column"]) else 0


if __name__ == "__main__":
    sys.exit(main())
