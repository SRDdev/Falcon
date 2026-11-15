#!/usr/bin/env bash
# AWS L4: once stage 2 (IP2P+SDEdit) and the guided/forgery tests are done, smoke-test FLUX.2-klein on the GPU
# (no CPU offload: the box has 15 GB RAM), then add it to stage 2, then run the generated-image track.
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/checkpoints/hf_home PY=$PWD/.venv/bin/python
while ps -eo args | grep -v grep | grep -qE "configs/stage2.yaml|queue_aws_extra.sh"; do sleep 60; done
if $PY scripts/dev/klein_smoke.py > klein_smoke.out 2>&1; then
  $PY run_experiment.py --config configs/stage2.yaml > stage2_klein.out 2>&1
else
  echo "klein smoke test failed; skipping FLUX.2" >> klein_smoke.out
fi
$PY run_experiment.py --config configs/gentrack.yaml > gentrack.out 2>&1
