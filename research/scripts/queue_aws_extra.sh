#!/usr/bin/env bash
# AWS L4 (after the vast.ai host went down): probes, guided test, then forgery test, alongside stage 2.
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/checkpoints/hf_home PY=$PWD/.venv/bin/python
$PY run_probes.py --config configs/probes.yaml > probes.out 2>&1 &
$PY run_experiment.py --config configs/guided_test.yaml > guided_test.out 2>&1
wait
$PY run_forgery_test.py --config configs/guided_test.yaml > forgery.out 2>&1
