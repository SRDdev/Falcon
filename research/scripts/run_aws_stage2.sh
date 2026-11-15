#!/usr/bin/env bash
# AWS L4 host: fetch models, run GPU smoke tests, then stage 2. Aborts if the smoke tests fail.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/checkpoints/hf_home PY=$PWD/.venv/bin/python
$PY -c "
from huggingface_hub import snapshot_download as s
for r in ['timbrooks/instruct-pix2pix','stable-diffusion-v1-5/stable-diffusion-v1-5']:
    s(r, allow_patterns=['*.json','*.txt','*fp16.safetensors','tokenizer/*'])"
$PY -m pytest -q -p no:cacheprovider 2>&1 | tail -3
$PY run_experiment.py --config configs/stage2.yaml
