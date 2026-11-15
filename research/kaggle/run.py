"""Kaggle GPU kernel: clone the branch, set up, run the 27-image baseline, then EWO + region + universality.
Outputs land in /kaggle/working/out (download with `kaggle kernels output`). Idempotent across reruns if
/kaggle/working persists within the session.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/prajak002/Falcon.git"
BRANCH = "peccavi-edit-spotlight"
WORK = Path("/kaggle/working")
ROOT = WORK / "Falcon" / "research"
OUT = WORK / "out"


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


if not ROOT.exists():
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORK / 'Falcon'}")
os.chdir(ROOT)
os.environ["HF_HOME"] = str(ROOT / "checkpoints" / "hf_home")
os.environ["PYTHONPATH"] = str(ROOT / "src")

# Kaggle images ship torch+CUDA; install everything else from requirements.txt.
reqs = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#") and not l.startswith(("torch", "torchvision"))]
sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{r}'" for r in reqs))

for d in ("logs", "external_repos", "checkpoints/wam"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
if not (ROOT / "external_repos/watermark-anything/.git").exists():
    sh("git clone --depth 1 https://github.com/facebookresearch/watermark-anything.git external_repos/watermark-anything")
with open(ROOT / "external_repos/VERSIONS.txt", "w") as f:
    rev = subprocess.check_output("git -C external_repos/watermark-anything rev-parse HEAD", shell=True, text=True).strip()
    f.write(f"watermark-anything https://github.com/facebookresearch/watermark-anything.git {rev}\n")
if not (ROOT / "checkpoints/wam/wam_mit.pth").exists():
    sh("wget -q -O checkpoints/wam/wam_mit.pth https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth")
shutil.copy(ROOT / "external_repos/watermark-anything/checkpoints/params.json", ROOT / "checkpoints/wam/params.json")

sh(f"{sys.executable} -c \"import torch;print('gpu', torch.cuda.get_device_name(0), torch.__version__)\"")
sh(f"{sys.executable} -m pytest -q -m 'not gpu' tests")

CFG = "configs/kaggle.yaml"
EXP = ROOT / "experiments" / "kaggle27"


def save_outputs():
    """Copy everything worth keeping, including the edited images (zipped), so a failed analysis never loses GPU work."""
    OUT.mkdir(exist_ok=True)
    for src in ("results", "ewo", "region", "prompts.jsonl", "config.yaml"):
        p = EXP / src
        if p.is_dir():
            shutil.copytree(p, OUT / p.name, dirs_exist_ok=True)
        elif p.exists():
            shutil.copy(p, OUT / p.name)
    for src in ("tables", "logs"):
        if (ROOT / src).is_dir():
            shutil.copytree(ROOT / src, OUT / src, dirs_exist_ok=True)
    for m in EXP.glob("manifest_*.json"):
        shutil.copy(m, OUT / m.name)
    if (EXP / "images").is_dir() and not (OUT / "images.zip").exists():
        shutil.make_archive(str(OUT / "images"), "zip", EXP / "images")
    print("outputs saved to", OUT, flush=True)


failures = []
try:
    sh(f"{sys.executable} run_experiment.py --config {CFG}")   # the GPU work; a failure here is fatal
finally:
    save_outputs()
for cmd in (f"{sys.executable} analysis/analyze.py --exp experiments/kaggle27",
            f"{sys.executable} analysis/universality.py --exp experiments/kaggle27",
            f"{sys.executable} run_ewo_analysis.py --config {CFG}",
            f"{sys.executable} run_region_analysis.py --config {CFG}"):
    try:
        sh(cmd)
    except subprocess.CalledProcessError as e:   # analyses are independent; keep going and report at the end
        failures.append(cmd)
        print("FAILED:", cmd, e, flush=True)
    save_outputs()
print("DONE; failed steps:", failures or "none", flush=True)
if failures:
    sys.exit(1)
