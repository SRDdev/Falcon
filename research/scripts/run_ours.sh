#!/usr/bin/env bash
# Falsification test of watermark-guided editing against baseline and re-embed (held-out images).
set -euo pipefail; cd "$(dirname "$0")/.."
python run_experiment.py --config configs/guided_test.yaml "$@"
