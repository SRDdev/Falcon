#!/usr/bin/env bash
# Stage-2 sequential benchmark: B0 (no watermark), B1 (watermark + editor), decode->edit->re-embed control.
set -euo pipefail; cd "$(dirname "$0")/.."
python run_experiment.py --config configs/stage2.yaml "$@"
