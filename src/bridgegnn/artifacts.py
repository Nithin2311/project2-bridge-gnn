"""Copy the aggregate (never patient-level) outputs into outputs/artifacts/ for the submission."""
from __future__ import annotations

import shutil
from pathlib import Path

# (glob relative to the outputs directory, destination subdirectory relative to the artifacts directory)
PATTERNS: list[tuple[str, str]] = [
    ("results/*.csv", "results"),
    ("eval/tables/*.csv", "eval/tables"),
    ("eval/tables/*.tex", "eval/tables"),
    ("eval/figures/*.pdf", "eval/figures"),
    ("layers/*.csv", "layers"),
    ("layers/tables/*.csv", "layers"),
    ("motifs/*.csv", "motifs"),
    ("motifs/*.pdf", "motifs"),
    ("pre2020/summary.csv", "pre2020"),
    ("pre2020/bridges.csv", "pre2020"),
    ("pre2020/target_definitions.csv", "pre2020"),
    ("pre2020/node_features.csv", "pre2020"),
    # step 02 writes nodes.csv, graph_edges.csv and construction_thresholds.json straight into outputs/artifacts/
    ("features/node_index.csv", ""),
    ("features/node_x.npy", ""),
    ("features/feature_groups.json", ""),
]
FORBIDDEN_SUFFIXES = {".parquet", ".txt"}


def _check_safe(path: Path) -> None:
    """Patient-level files never enter the artifact tree: parquet (rows, predictions) and txt (subsample lists)."""
    if path.suffix in FORBIDDEN_SUFFIXES:
        raise ValueError(f"refusing to collect a patient-level file type: {path}")


COLLECTED_SUBDIRS = sorted({sub for _, sub in PATTERNS if sub})


def collect(out: Path, dest: Path) -> list[Path]:
    """Copy every file matching PATTERNS from `out` into `dest`; missing sources are skipped (partial trees are legal).

    The collected subdirectories of `dest` are emptied first so a file the current run did not produce cannot survive
    from an earlier run or from the git checkout; top-level files (written directly by step 02) are left alone.
    """
    out, dest = Path(out), Path(dest)
    for sub in COLLECTED_SUBDIRS:
        if (dest / sub).exists():
            shutil.rmtree(dest / sub)
    copied = []
    seen: dict[Path, Path] = {}
    for pattern, sub in PATTERNS:
        for src in sorted(out.glob(pattern)):
            if not src.is_file():
                continue
            _check_safe(src)
            target = dest / sub / src.name
            if target in seen:
                raise ValueError(f"two sources for one artifact {target.name}: {seen[target]} and {src}")
            seen[target] = src
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            copied.append(target)
    return copied
