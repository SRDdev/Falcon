#!/usr/bin/env bash
# AWS L4: after stage 2 finishes, run the generated-image track (Tree-Ring, Gaussian Shading).
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/checkpoints/hf_home
while ps -eo args | grep -v grep | grep -q "configs/stage2.yaml"; do sleep 60; done
.venv/bin/python run_experiment.py --config configs/gentrack.yaml > gentrack.out 2>&1
