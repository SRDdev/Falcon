"""Post-hoc watermark wrappers with a uniform interface.

Every watermark exposes
  embed(img, bits)            -> watermarked PIL image          (official implementation)
  decode(img)                 -> np.ndarray of {0,1} bits         (official implementation)
  bit_logits(x01) [optional]  -> differentiable per-bit logits    (used only by guided editing)

Detection decision: a k-bit payload is "detected" when the number of matching bits m satisfies
P[Binomial(k, 1/2) >= m] <= target_fpr, i.e. a one-sided test against a decoder emitting fair coin
flips on unwatermarked images. That null assumption is checked empirically (clean-image FPR) in the
calibration stage; we report both.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binom

from .utils import RESEARCH_ROOT, get_device, pil_to_tensor, tensor_to_pil

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def detection_threshold(nbits: int, target_fpr: float) -> int:
    """Smallest number of matching bits m with P[Bin(nbits,.5) >= m] <= target_fpr."""
    for m in range(nbits + 1):
        if binom.sf(m - 1, nbits, 0.5) <= target_fpr:
            return m
    return nbits + 1


def p_value(matches: int, nbits: int) -> float:
    return float(binom.sf(matches - 1, nbits, 0.5))


class Watermark:
    name: str = "base"
    nbits: int = 0
    differentiable: bool = False

    def embed(self, img: Image.Image, bits: np.ndarray) -> Image.Image:
        raise NotImplementedError

    def decode(self, img: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def bit_logits(self, x01: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} has no differentiable decoder")

    def score(self, img: Image.Image, bits: np.ndarray, target_fpr: float) -> dict:
        pred = self.decode(img)
        matches = int((pred == bits).sum())
        thr = detection_threshold(self.nbits, target_fpr)
        return {
            "bit_acc": matches / self.nbits,
            "matches": matches,
            "p_value": p_value(matches, self.nbits),
            "detected": bool(matches >= thr),
            "threshold_matches": thr,
            "exact_payload": bool(matches == self.nbits),
        }


@contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class WAM(Watermark):
    """Watermark Anything (Sander et al., ICLR 2025), MIT checkpoint. Full-image embedding."""
    name = "wam"
    nbits = 32
    differentiable = True

    def __init__(self, ckpt_dir: str | None = None, device=None, scaling_w: float | None = None):
        repo = RESEARCH_ROOT / "external_repos" / "watermark-anything"
        ckpt_dir = Path(ckpt_dir or RESEARCH_ROOT / "checkpoints" / "wam")
        sys.path.insert(0, str(repo))
        from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
        from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402

        self._msg_predict = msg_predict_inference
        self.device = device or get_device()
        with _cwd(repo):  # params.json refers to configs/*.yaml relative to the repo
            self.model = load_model_from_checkpoint(str(ckpt_dir / "params.json"),
                                                    str(ckpt_dir / "wam_mit.pth")).to(self.device).eval()
        if scaling_w is not None:  # embedding strength (ablation); default from checkpoint = 2.0
            self.model.scaling_w = scaling_w
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _norm(self, x01):
        return (x01 - IMAGENET_MEAN.to(x01)) / IMAGENET_STD.to(x01)

    def _unnorm(self, x):
        return x * IMAGENET_STD.to(x) + IMAGENET_MEAN.to(x)

    @torch.no_grad()
    def embed(self, img, bits):
        x = self._norm(pil_to_tensor(img, self.device))
        msg = torch.as_tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.model.embed(x, msg)["imgs_w"]
        return tensor_to_pil(self._unnorm(out))

    @torch.no_grad()
    def detect_raw(self, img):
        preds = self.model.detect(self._norm(pil_to_tensor(img, self.device)))["preds"]
        return torch.sigmoid(preds[:, 0:1]), preds[:, 1:]

    @torch.no_grad()
    def decode(self, img):
        mask, bitp = self.detect_raw(img)
        return self._msg_predict(bitp, mask).cpu().numpy()[0].astype(np.uint8)

    def mask_fraction(self, img) -> float:
        mask, _ = self.detect_raw(img)
        return float((mask > 0.5).float().mean())

    def bit_logits(self, x01):
        preds = self.model.detect(self._norm(x01))["preds"]
        mask = torch.sigmoid(preds[:, 0:1])
        return (preds[:, 1:] * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1e-6)


class TrustMarkWM(Watermark):
    """TrustMark (Bui et al., 2023), Adobe, MIT. Raw 100-bit payload (ECC disabled so bit accuracy
    is measured on the channel itself, comparable to the other methods)."""
    name = "trustmark"
    nbits = 100
    differentiable = True

    def __init__(self, model_type: str = "Q", device=None, strength: float = 1.0):
        from trustmark import TrustMark

        self.device = device or get_device()
        self.tm = TrustMark(verbose=False, model_type=model_type, use_ECC=False, secret_len=100,
                            device=str(self.device), loadRemover=False)
        self.strength = strength
        for p in self.tm.decoder.parameters():
            p.requires_grad_(False)

    def embed(self, img, bits):
        return self.tm.encode(img, "".join(str(int(b)) for b in bits), MODE="binary",
                              WM_STRENGTH=self.strength).convert("RGB")

    def decode(self, img):
        s, _, _ = self.tm.decode(img, MODE="binary")
        return np.array([int(c) for c in s], dtype=np.uint8)

    def bit_logits(self, x01):
        scale = self.tm.concentrate_wm_region
        _, _, h, w = x01.shape
        ch, cw = int(h * scale), int(w * scale)
        top, left = (h - ch) // 2, (w - cw) // 2
        x = x01[:, :, top:top + ch, left:left + cw]
        r = self.tm.model_resolution_dec
        x = F.interpolate(x, size=(r, r), mode="bilinear", antialias=True, align_corners=False)
        return self.tm.decoder.decoder(x * 2 - 1)


class DWTDCT(Watermark):
    """DWT-DCT(-SVD) via `invisible-watermark` (the scheme used by the Stable Diffusion reference code).
    Not differentiable: evaluated as a baseline only."""
    name = "dwtdct"
    nbits = 32

    def __init__(self, mode: str = "dwtDctSvd", **_):  # plain dwtDct: 52% L0 detection (SETUP.md)
        self.mode = mode
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        self._enc, self._dec = WatermarkEncoder, WatermarkDecoder

    def embed(self, img, bits):
        import cv2

        enc = self._enc()
        enc.set_watermark("bits", [int(b) for b in bits])
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        out = enc.encode(bgr, self.mode)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def decode(self, img):
        import cv2

        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return np.array(self._dec("bits", self.nbits).decode(bgr, self.mode), dtype=np.uint8)


class SaliencyWAM(WAM):
    """WAM enhanced with 64x64 dual-zone saliency modulation, FFT mid-band locking, and tile-voted decoding."""
    name = "wam_saliency"
    nbits = 32
    differentiable = True

    def __init__(
        self,
        ckpt_dir: str | None = None,
        device=None,
        scaling_w: float | None = None,
        grid: tuple[int, int] = (8, 8),
        k_fg: int = 4,
        k_bg: int = 12,
        top_k_tiles: int = 16,
        spectral_blend: float = 0.35,
        band: tuple[float, float] = (3.0, 16.0),
        saliency_backend: str = "auto",
    ):
        super().__init__(ckpt_dir=ckpt_dir, device=device, scaling_w=scaling_w)
        from .saliency import SaliencyExtractor
        self.grid = grid
        self.k_fg = k_fg
        self.k_bg = k_bg
        self.top_k_tiles = top_k_tiles
        self.spectral_blend = spectral_blend
        self.band = band
        self.saliency = SaliencyExtractor(backend=saliency_backend)

    def _bandpass_tiles(self, resid: np.ndarray) -> np.ndarray:
        """Apply 2D FFT bandpass filtering per 64x64 tile to concentrate energy in VAE-resilient mid-bands."""
        from .erasure import grid_boxes
        h, w = resid.shape[:2]
        out = np.zeros_like(resid)
        for y0, x0, y1, x1 in grid_boxes(h, w, *self.grid):
            th, tw = y1 - y0, x1 - x0
            fy = np.fft.fftfreq(th)[:, None] * th
            fx = np.fft.fftfreq(tw)[None, :] * tw
            r = np.sqrt(fy * fy + fx * fx)
            mask = ((r >= self.band[0]) & (r <= self.band[1])).astype(np.float32)
            for c in range(resid.shape[2]):
                patch = resid[y0:y1, x0:x1, c]
                bp = np.real(np.fft.ifft2(np.fft.fft2(patch) * mask))
                out[y0:y1, x0:x1, c] = bp
        return out

    @torch.no_grad()
    def embed(self, img: Image.Image, bits: np.ndarray) -> Image.Image:
        from .saliency import dual_zone_tile_weights
        wm_raw = super().embed(img, bits)
        orig = np.asarray(img.convert("RGB"), dtype=np.float32)
        resid = np.asarray(wm_raw, dtype=np.float32) - orig
        if self.spectral_blend > 0:
            bp = self._bandpass_tiles(resid)
            bp_norm = np.sqrt(np.mean(resid ** 2) / (np.mean(bp ** 2) + 1e-12))
            resid = (1.0 - self.spectral_blend) * resid + self.spectral_blend * (bp * bp_norm)
        w_map, _ = dual_zone_tile_weights(
            img, grid=self.grid, k_fg=self.k_fg, k_bg=self.k_bg, extractor=self.saliency,
        )
        out = np.clip(orig + resid * w_map[..., None], 0, 255).round().astype(np.uint8)
        return Image.fromarray(out)

    @torch.no_grad()
    def decode(self, img: Image.Image) -> np.ndarray:
        from .erasure import grid_boxes
        mask, bitp = self.detect_raw(img)
        _, _, h, w = bitp.shape
        boxes = grid_boxes(h, w, *self.grid)
        tile_conf = []
        tile_logits = []
        for y0, x0, y1, x1 in boxes:
            m_tile = mask[0, 0, y0:y1, x0:x1]
            b_tile = bitp[0, :, y0:y1, x0:x1]
            conf = float(m_tile.mean() * b_tile.abs().mean())
            # Mask-weighted bit logits inside this 64x64 tile
            logit = (b_tile * m_tile).sum((1, 2)) / m_tile.sum().clamp_min(1e-6)
            tile_conf.append(conf)
            tile_logits.append(logit)
        order = np.argsort(-np.array(tile_conf))[: min(self.top_k_tiles, len(boxes))]
        weights = torch.tensor([tile_conf[i] for i in order], device=bitp.device, dtype=bitp.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        stacked = torch.stack([tile_logits[i] for i in order], dim=0)
        agg = (stacked * weights[:, None]).sum(dim=0)
        return (agg > 0).cpu().numpy().astype(np.uint8)


REGISTRY = {"wam": WAM, "wam_saliency": SaliencyWAM, "trustmark": TrustMarkWM, "dwtdct": DWTDCT}


def build_watermark(name: str, **kwargs) -> Watermark:
    if name == "peccavi_edit":  # numpy-only module, kept torch-free
        from .peccavi_edit import PeccaviEdit

        return PeccaviEdit(**kwargs)
    if name == "peccavi_hybrid":  # numpy-only module, kept torch-free
        from .peccavi_edit import PeccaviHybrid

        return PeccaviHybrid(**kwargs)
    return REGISTRY[name](**kwargs)


class VINE(Watermark):
    """VINE-B/R (Lu et al., ICLR 2025). 100-bit payload; encoder works at 256x256 and the residual is upsampled to
    full resolution (the repo's own scripts). Weights: Shilin-LU/VINE-{B,R}-{Enc,Dec}; needs SD-Turbo at load.
    Licence: NTUITIVE non-commercial (GitHub LICENSE; the HF cards say MIT - record both). CUDA only.
    Install into a separate env: pip install git+https://github.com/Shilin-LU/VINE (pins torch 2.0.1)."""
    name = "vine"
    nbits = 100

    def __init__(self, variant: str = "R", device=None):
        from torchvision import transforms as T
        from vine.src.stega_encoder_decoder import CustomConvNeXt
        from vine.src.vine_turbo import VINE_Turbo

        self.device = device or get_device()
        self.enc = VINE_Turbo.from_pretrained(f"Shilin-LU/VINE-{variant}-Enc").to(self.device).eval()
        self.dec = CustomConvNeXt.from_pretrained(f"Shilin-LU/VINE-{variant}-Dec").to(self.device).eval()
        self.T, self.bic = T, T.InterpolationMode.BICUBIC

    @torch.no_grad()
    def embed(self, img, bits):
        T = self.T
        img = img.convert("RGB")
        full = T.ToTensor()(img).unsqueeze(0).to(self.device) * 2 - 1
        small = T.Compose([T.Resize(256, interpolation=self.bic), T.ToTensor()])(img).unsqueeze(0).to(self.device) * 2 - 1
        msg = torch.tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
        res = T.Resize(img.size[::-1], interpolation=self.bic)(self.enc(small, msg) - small)
        return T.ToPILImage()(((res + full) * 0.5 + 0.5).clamp(0, 1)[0].cpu())

    @torch.no_grad()
    def decode(self, img):
        T = self.T
        x = T.Compose([T.Resize(256, interpolation=self.bic), T.ToTensor()])(img.convert("RGB")).unsqueeze(0).to(self.device)
        return np.round(self.dec(x)[0].cpu().numpy()).clip(0, 1).astype(np.uint8)


class RobustWide(Watermark):
    """Robust-Wide (Hu et al., ECCV 2024), MIT. 64-bit payload at 512x512, inputs normalised to [-1,1], bit decision
    > 0.5 (the repo's utils). Checkpoints are distributed via Google Drive only: put wm_model_config.yaml and
    wm_model.ckpt in external_repos/Robust-Wide/checkpoints/. Repo root must be importable (model.py)."""
    name = "robustwide"
    nbits = 64

    def __init__(self, repo_dir: str | None = None, device=None):
        from omegaconf import OmegaConf
        from torchvision import transforms as T

        repo = Path(repo_dir or RESEARCH_ROOT / "external_repos" / "Robust-Wide")
        sys.path.insert(0, str(repo))
        from model import WatermarkModel  # noqa: E402

        self.device = device or get_device()
        cfg = OmegaConf.load(repo / "checkpoints" / "wm_model_config.yaml")
        self.m = WatermarkModel(**cfg)
        self.m.load_state_dict(torch.load(repo / "checkpoints" / "wm_model.ckpt", map_location="cpu"))
        self.m = self.m.eval().to(self.device)
        self.prep = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512),
                               T.ToTensor(), T.Normalize([0.5], [0.5])])
        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    def embed(self, img, bits):
        x = self.prep(img.convert("RGB")).unsqueeze(0).to(self.device)
        msg = torch.tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.to_pil((self.m.encoder(x, msg) / 2 + 0.5).clamp(0, 1)[0].cpu())

    @torch.no_grad()
    def decode(self, img):
        x = self.prep(img.convert("RGB")).unsqueeze(0).to(self.device)
        return (self.m.decoder(x)[0].cpu().numpy() > 0.5).astype(np.uint8)


REGISTRY.update({"vine": VINE, "robustwide": RobustWide})
