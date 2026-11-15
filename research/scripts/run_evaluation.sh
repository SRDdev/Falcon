#!/usr/bin/env bash
# Re-run detection + metrics + statistics for an experiment (no editing). Usage: run_evaluation.sh [stage2]
set -euo pipefail; cd "$(dirname "$0")/.."
EXP="${1:-stage2}"
python run_experiment.py --config "configs/${EXP}.yaml" --stages evaluate
python analysis/analyze.py --exp "experiments/${EXP}"
