#!/usr/bin/env bash
# Environment setup for the GPU host (vast.ai RTX 3060 image, /venv/main, CUDA 12.8 driver).
# Idempotent: safe to re-run. Logs to research/logs/setup.log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs external_repos checkpoints
# Two supported hosts:
#  vast.ai image:  /venv/main exists -> install torch cu128 into it
#  elsewhere (AWS DL AMI): isolated uv venv in .venv (system env untouched)
# Both hosts pin the same torch build so results differ only by GPU.
TORCH="torch==2.11.0 torchvision==0.26.0"
if [ -d /venv/main ]; then
  PY=/venv/main/bin/python
else  # isolated venv (no system site-packages: the AMI's binary wheels clash)
  [ -x .venv/bin/python ] || uv venv --python 3.12 .venv
  PY=$ROOT/.venv/bin/python
fi
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu128 $TORCH
export HF_HOME="${HF_HOME:-$ROOT/checkpoints/hf_home}"
uv pip install --python $PY -r requirements.txt

# --- External repos (pinned commits recorded in external_repos/VERSIONS.txt) ---
clone() {  # url dir
  [ -d "external_repos/$2/.git" ] || git clone --depth 1 "$1" "external_repos/$2"
}
clone https://github.com/facebookresearch/watermark-anything.git watermark-anything
clone https://github.com/SRDdev/Falcon.git Falcon
: > external_repos/VERSIONS.txt
for d in external_repos/*/; do
  echo "$(basename "$d") $(git -C "$d" remote get-url origin) $(git -C "$d" rev-parse HEAD)" >> external_repos/VERSIONS.txt
done

# --- WAM checkpoint (MIT-licensed variant) ---
mkdir -p checkpoints/wam
[ -s checkpoints/wam/wam_mit.pth ] || wget -q -O checkpoints/wam/wam_mit.pth \
  https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth
cp -n external_repos/watermark-anything/checkpoints/params.json checkpoints/wam/ 2>/dev/null || true

$PY - <<'EOF'
import torch, diffusers, transformers, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "gpu", torch.cuda.get_device_name(0), "diffusers", diffusers.__version__,
      "transformers", transformers.__version__)
EOF
echo SETUP_OK
