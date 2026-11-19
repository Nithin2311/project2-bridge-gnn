#!/usr/bin/env bash
# Reproduce every Project 2 artifact from a fresh clone: steps 00-09 in dependency order, the experiment grid through
# the step 07 driver, evaluation, report fragments and macros, then the aggregate artifact copy.
#
#   ./scripts/run_all.sh             full run (~9 h on an M4: grid ~6 h, baselines ~1.5 h, evaluation < 15 min)
#   ./scripts/run_all.sh --from 06   resume at a label (labels: 00 01 02 03 04 05 06 07 09 08 08b collect macros)
#   ./scripts/run_all.sh --check     tests, grid dry run and input check only (~2 min)
#
# Every step appends START/END lines with timestamps, elapsed seconds and exit code to outputs/logs/run_all.log.
# The first failure stops the script and names the label; rerun with --from <label> after fixing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
LOG=outputs/logs/run_all.log
FROM=""
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
LABELS=(00 01 02 03 04 05 06 07 09 08 08b collect macros)
echo "RUN $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) args=$*" | tee -a "$LOG"
STARTED=0
[ -z "$FROM" ] && STARTED=1

run_step() {
  local label="$1"; shift
  if [ "$STARTED" -eq 0 ]; then
    if [ "$label" = "$FROM" ]; then STARTED=1; else echo "skip $label"; return 0; fi
  fi
  local t0 now
  t0=$(date +%s); now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "START $label $now  $*" | tee -a "$LOG"
  local code=0
  "$@" || code=$?
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "END $label $now elapsed=$(( $(date +%s) - t0 ))s exit=$code" | tee -a "$LOG"
  if [ "$code" -ne 0 ]; then echo "run_all: step $label failed (exit $code); resume with --from $label" >&2; exit "$code"; fi
}

if [ "$CHECK" -eq 1 ]; then
  run_step check-tests uv run pytest -q
  run_step check-grid uv run python scripts/07_experiments.py --dry-run
  run_step check-inputs uv run python scripts/00_check_inputs.py
  exit 0
fi

if [ -n "$FROM" ] && [[ ! " ${LABELS[*]} " =~ " $FROM " ]]; then
  echo "unknown label $FROM; labels: ${LABELS[*]}" >&2; exit 2
fi

# GNN experiment ids other than the main comparison, read from the grid so the list never goes stale
GRID_IDS=$(uv run python -c "import yaml;g=yaml.safe_load(open('config/experiments.yaml'));print(' '.join(e['id'] for e in g['experiments'] if e['base']=='gnn' and not e['id'].startswith('G-')))")

run_step 00 uv run python scripts/00_check_inputs.py
run_step 01 uv run python scripts/01_rows.py
run_step 02 uv run python scripts/02_features.py
run_step 03 uv run python scripts/03_directed_layer.py
run_step 04 uv run python scripts/04_motifs.py
run_step 05 uv run python scripts/07_experiments.py --only-baselines
run_step 06 uv run python scripts/07_experiments.py --ids G-gcn G-sage G-gat
# shellcheck disable=SC2086
run_step 07 uv run python scripts/07_experiments.py --ids $GRID_IDS
run_step 09 uv run python scripts/09_pre2020_graph.py
run_step 08 uv run python scripts/08_evaluate.py
run_step 08b uv run python scripts/08_report_baselines.py
run_step collect uv run python scripts/collect_artifacts.py
run_step macros uv run python scripts/report_macros.py   # reads outputs/artifacts, so after collect
echo "run_all: done" | tee -a "$LOG"
