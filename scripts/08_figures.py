"""Step 8f: the report's main figure per primary target (curated models, three panels). Cheap: no bootstraps.
Rerunnable: `uv run python scripts/08_figures.py [--runs G-gat G-sage B-cov_lr B-hgb_all B-prevalence]`.
"""
from __future__ import annotations

import argparse

from bridgegnn import config
from bridgegnn.eval import figures, report


def main() -> None:
    cfg = config.load()
    out = config.outputs_dir(cfg)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=figures.REPORT_RUNS)
    ap.add_argument("--targets", nargs="+", default=list(cfg["task"]["primary_targets"]) + list(cfg["task"]["secondary_targets"]))
    args = ap.parse_args()
    preds = report.load_predictions(out / "predictions")
    for target in args.targets:
        path = out / "eval" / "figures" / f"fig_main_{target}.pdf"
        drawn = figures.main_figure(preds, target, args.runs, path)
        print(f"{path.name}: {', '.join(drawn) if drawn else 'nothing drawn'}")


if __name__ == "__main__":
    main()
