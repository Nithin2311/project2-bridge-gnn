import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "run_all.sh").read_text()


def test_collect_runs_before_macros():
    """report_macros reads outputs/artifacts, so the artifact copy must precede it (else macros come from the committed tree)."""
    labels = re.findall(r"^run_step (\S+) ", SCRIPT, flags=re.MULTILINE)
    assert labels.index("collect") < labels.index("macros")
    assert labels.index("08b") < labels.index("collect")
    assert labels.index("01") < labels.index("10") < labels.index("08")


def test_run_all_logs_the_commit():
    assert "git rev-parse HEAD" in SCRIPT
