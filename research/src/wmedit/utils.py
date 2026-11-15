"""Shared helpers: seeding, image conversion, environment manifest."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

RESEARCH_ROOT = Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def pil_to_tensor(img: Image.Image, device=None) -> torch.Tensor:
    """PIL RGB -> float tensor (1,3,H,W) in [0,1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device) if device is not None else t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) or (3,H,W) in [0,1] -> PIL (quantised to uint8, as a real pipeline would)."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def environment_manifest() -> dict:
    import importlib.metadata as md

    pkgs = {}
    for name in ["torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "scipy",
                 "lpips", "open_clip_torch", "trustmark", "invisible-watermark", "datasets", "pandas"]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(get_device())
    ext = RESEARCH_ROOT / "external_repos" / "VERSIONS.txt"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "packages": pkgs,
        "code_sha256": code_fingerprint(),
        "git_commit": _cmd(["git", "-C", str(RESEARCH_ROOT), "rev-parse", "HEAD"]),
        "external_repos": ext.read_text().splitlines() if ext.exists() else [],
    }


def code_fingerprint() -> str:
    """Hash of all source files, so results can be tied to an exact code state even without git."""
    h = hashlib.sha256()
    for p in sorted((RESEARCH_ROOT / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
