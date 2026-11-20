import polars as pl

from bridgegnn.eval import figures
from tests.test_eval_report import make_preds


def test_main_figure_draws_three_panels_for_the_curated_runs(tmp_path):
    preds = pl.concat([make_preds("G-gat", "gat", [0, 1], seed=0), make_preds("B-cov_lr", "cov_lr", [0], seed=1),
                       make_preds("B-prevalence", "prevalence", [0], seed=2).with_columns(pl.lit(0.2).alias("p"))])
    out = tmp_path / "fig_main_mental_health.pdf"
    drawn = figures.main_figure(preds, "mental_health", ["G-gat", "G-sage", "B-cov_lr", "B-prevalence"], out)
    assert out.exists() and out.stat().st_size > 1000
    assert drawn == ["G-gat", "B-cov_lr", "B-prevalence"]   # G-sage absent from the predictions is skipped, not an error
