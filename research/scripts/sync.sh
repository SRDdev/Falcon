#!/usr/bin/env bash
# push code to / pull results from the GPU host. Usage: scripts/sync.sh push|pull
set -euo pipefail
# Hosts: default AWS L4; WM_TARGET=vast for the (decommissioned) vast.ai box.
if [ "${WM_TARGET:-aws}" = aws ]; then  # vast.ai host decommissioned 2026-10-01
  source "$HOME/Downloads/ocr-rnd/ocr-rnd.env"
  HOST="ubuntu@$OCR_RND_EIP"; REMOTE="/home/ubuntu/wm_research/"
  SSH="ssh -i $HOME/Downloads/ocr-rnd/ocr-rnd.pem -o BatchMode=yes"
else
  HOST="${WM_HOST:-root@185.62.108.226}"; REMOTE="/workspace/research/"
  SSH="ssh -p ${WM_PORT:-45701} -o BatchMode=yes"
fi
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/"
case "$1" in
  push) rsync -az -e "$SSH" --exclude experiments/ --exclude checkpoints/ --exclude datasets/hf_cache \
          --exclude external_repos/ --exclude .venv/ --exclude '__pycache__' "$LOCAL" "$HOST:$REMOTE" ;;
  pull) rsync -az -e "$SSH" --include '*/' --include '*.json' --include '*.jsonl' --include '*.csv' \
          --include '*.yaml' --include '*.log' --include '*.png' --include '*.txt' --exclude '*' \
          "$HOST:${REMOTE}experiments/" "${LOCAL}experiments/"
        rsync -az -e "$SSH" "$HOST:${REMOTE}external_repos/VERSIONS.txt" "${LOCAL}external_repos/" ;;
esac
