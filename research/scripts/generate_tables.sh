#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
for e in pilot guided_test stage2; do
  [ -f "experiments/$e/results/rows.csv" ] && python analysis/analyze.py --exp "experiments/$e"
done
