"""Fast saliency extraction and 64x64 dual-zone tile partitioning for edit-resilient watermarking.

Supports:
  - TinyHD backend (Hu et al., WACV 2023): lightweight distilled saliency network (~15 ms/image)
    when PyTorch and TinyHD weights are available under `external_repos/tinyHD`.
  - Spectral Residual backend (Hou & Zhang, CVPR 2007): pure NumPy/SciPy fallback (< 5 ms CPU)
    requiring no external weights, used automatically when TinyHD weights are absent or on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from .erasure import grid_boxes
from .utils import RESEARCH_ROOT

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def spectral_saliency(img: Image.Image, sigma: float = 8.0) -> np.ndarray:
    """Fast Spectral Residual saliency map in [0, 1] (pure NumPy/SciPy, < 5 ms)."""
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = x @ LUMA
    h, w = gray.shape
    # Downsample to 128x128 for speed and scale invariance
    small = np.asarray(
        Image.fromarray((gray * 255).clip(0, 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    residual = log_amp - uniform_filter(log_amp, size=3)
    sal_small = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal_small = gaussian_filter(sal_small, sigma=2.5)
    sal = np.asarray(
        Image.fromarray(
            ((sal_small - sal_small.min()) / (sal_small.max() - sal_small.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    if sigma > 0:
        sal = gaussian_filter(sal, sigma=sigma)
    lo, hi = sal.min(), sal.max()
    return ((sal - lo) / (hi - lo + 1e-8)).astype(np.float32)


def texture_energy_map(gray: np.ndarray, size: int = 9) -> np.ndarray:
    """Normalized local contrast / texture energy in [0, 1]."""
    mu = uniform_filter(gray, size)
    var = np.clip(uniform_filter(gray * gray, size) - mu * mu, 0.0, None)
    sd = np.sqrt(var)
    norm = np.percentile(sd, 95) + 1e-6
    return np.clip(sd / norm, 0.0, 1.0).astype(np.float32)


class SaliencyExtractor:
    """Unified fast saliency extractor with TinyHD and Spectral Residual backends."""

    def __init__(self, backend: str = "auto", weights_path: str | None = None, device: str | None = None):
        self.requested_backend = backend
        self.weights_path = Path(weights_path) if weights_path else (
            RESEARCH_ROOT / "external_repos" / "tinyHD" / "weights" / "d1d2d3_S_lt.pth"
        )
        self._device = device
        self._model = None
        self.active_backend = self._init_backend(backend)

    def _init_backend(self, backend: str) -> str:
        if backend == "spectral":
            return "spectral"
        repo = RESEARCH_ROOT / "external_repos" / "tinyHD"
        if backend in ("auto", "tinyhd") and repo.exists() and self.weights_path.exists():
            try:
                import torch  # noqa: F401
                sys.path.insert(0, str(repo / "src"))
                # Attempt lazy TinyHD load if repo is cloned and weights exist
                self._load_tinyhd()
                return "tinyhd"
            except Exception:
                if backend == "tinyhd":
                    raise
        return "spectral"

    def _load_tinyhd(self):
        import torch
        from torchvision import transforms

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        ckpt = torch.load(self.weights_path, map_location=dev)
        if hasattr(ckpt, "eval"):
            self._model = ckpt.to(dev).eval()
        else:
            raise RuntimeError("TinyHD checkpoint requires model class instantiation from tinyHD/src")
        self._tf = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image) -> np.ndarray:
        if self.active_backend == "tinyhd" and self._model is not None:
            import torch

            w, h = img.size
            x = self._tf(img.convert("RGB")).unsqueeze(0).to(self._device)
            with torch.no_grad():
                pred = self._model(x)
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
                sal = torch.sigmoid(pred).squeeze().cpu().numpy()
            sal_img = Image.fromarray((sal * 255).clip(0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            arr = np.asarray(sal_img, dtype=np.float32) / 255.0
            return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8)).astype(np.float32)
        return spectral_saliency(img)


def tile_scores(
    img: Image.Image,
    grid: tuple[int, int] = (8, 8),
    extractor: SaliencyExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute boxes, saliency map, per-tile saliency, and per-tile texture energy."""
    extractor = extractor or SaliencyExtractor(backend="auto")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    gray = rgb @ LUMA
    sal_map = extractor(img)
    tex_map = texture_energy_map(gray)
    boxes = grid_boxes(h, w, *grid)

    tile_sal = np.array([sal_map[y0:y1, x0:x1].mean() for y0, x0, y1, x1 in boxes], dtype=np.float32)
    tile_tex = np.array([tex_map[y0:y1, x0:x1].mean() for y0, x0, y1, x1 in boxes], dtype=np.float32)
    return boxes, sal_map, tile_sal, tile_tex


def bucketed_tile_selection(
    img: Image.Image,
    grid: tuple[int, int] = (8, 8),
    bucket_shape: tuple[int, int] = (2, 2),
    fg_buckets: int = 4,
    extractor: SaliencyExtractor | None = None,
) -> dict:
    """Partition an 8x8 grid of 64x64 tiles into 16 spatial buckets of 4 tiles each.

    In the `fg_buckets` most salient buckets, selects the strongest foreground-textured tile.
    In the remaining `16 - fg_buckets` buckets, selects the strongest background-textured tile.
    This allows 100% blind detection by scanning only the 4 candidates per bucket.
    """
    rows, cols = grid
    br, bc = bucket_shape
    if rows % br != 0 or cols % bc != 0:
        raise ValueError(f"grid {grid} must be divisible by bucket_shape {bucket_shape}")

    boxes, sal_map, tile_sal, tile_tex = tile_scores(img, grid=grid, extractor=extractor)
    b_rows, b_cols = rows // br, cols // bc
    n_buckets = b_rows * b_cols

    buckets: list[list[int]] = []
    bucket_sal = np.zeros(n_buckets, dtype=np.float32)
    for bi in range(b_rows):
        for bj in range(b_cols):
            members = [
                (bi * br + dr) * cols + (bj * bc + dc)
                for dr in range(br)
                for dc in range(bc)
            ]
            buckets.append(members)
            bucket_sal[len(buckets) - 1] = float(np.mean(tile_sal[members]))

    fg_order = np.argsort(-bucket_sal)
    fg_set = set(int(x) for x in fg_order[:fg_buckets])

    chosen_idx: list[int] = []
    zone_labels: list[str] = []
    for b_id, members in enumerate(buckets):
        m_arr = np.array(members)
        if b_id in fg_set:
            # Foreground anchor: high saliency + high texture
            sc = (tile_sal[m_arr] + 0.1) * (tile_tex[m_arr] + 0.2)
            zone_labels.append("fg")
        else:
            # Background anchor: low saliency + high texture
            sc = (1.0 - tile_sal[m_arr] + 0.1) * (tile_tex[m_arr] + 0.2)
            zone_labels.append("bg")
        best_local = int(np.argmax(sc))
        chosen_idx.append(int(m_arr[best_local]))

    return {
        "boxes": boxes,
        "buckets": buckets,
        "chosen_idx": np.array(chosen_idx, dtype=int),
        "chosen_boxes": boxes[chosen_idx],
        "zone_labels": zone_labels,
        "sal_map": sal_map,
        "tile_sal": tile_sal,
        "tile_tex": tile_tex,
    }


def dual_zone_tile_weights(
    img: Image.Image,
    grid: tuple[int, int] = (8, 8),
    k_fg: int = 4,
    k_bg: int = 12,
    fg_weight: float = 0.9,
    bg_weight: float = 1.25,
    other_weight: float = 0.55,
    extractor: SaliencyExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a smooth spatial weight mask [H, W] over 64x64 tiles for WAM modulation."""
    boxes, _, tile_sal, tile_tex = tile_scores(img, grid=grid, extractor=extractor)
    rgb_h, rgb_w = img.size[1], img.size[0]

    fg_score = (tile_sal + 0.05) * (tile_tex + 0.15)
    bg_score = (1.0 - tile_sal + 0.05) * (tile_tex + 0.15)

    fg_idx = set(int(i) for i in np.argsort(-fg_score)[:k_fg])
    bg_candidates = [int(i) for i in np.argsort(-bg_score) if int(i) not in fg_idx]
    bg_idx = set(bg_candidates[:k_bg])

    weights = np.full(len(boxes), other_weight, dtype=np.float32)
    for i in fg_idx:
        weights[i] = fg_weight
    for i in bg_idx:
        weights[i] = bg_weight

    w_map = np.zeros((rgb_h, rgb_w), dtype=np.float32)
    for i, (y0, x0, y1, x1) in enumerate(boxes):
        w_map[y0:y1, x0:x1] = weights[i]
    w_map = gaussian_filter(w_map, sigma=4.0)
    w_map = w_map / (w_map.mean() + 1e-8)
    return w_map.astype(np.float32), weights
