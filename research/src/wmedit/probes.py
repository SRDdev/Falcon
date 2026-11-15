"""Failure-mode probes: where along the editing pipeline is the watermark lost?

For each watermarked L0 image we apply a ladder of increasingly strong transformations that share
components with the real editors, and decode after each:

  vae_roundtrip           SD-1.5 VAE encode -> decode (the latent bottleneck alone, no diffusion)
  ip2p_identity           InstructPix2Pix with a no-op instruction ("keep the image unchanged")
  ip2p_imgcfg_<s>         a real edit at image-guidance s (higher = closer to the input)
  ip2p_steps_<n>          a real edit with n denoising steps
  sdedit_<strength>       SDEdit at noise strength in {0.1 .. 0.7}, source caption (no semantic change)

Residual survival: with r0 = L0 - original and rT = T(L0) - T(original) under the same seed, we
report the correlation corr(r0, rT) and the fraction of r0's energy surviving per radial frequency
band. This separates "watermark energy destroyed" from "decoder confused by changed content".

Region analysis (WAM only, localised detector): fraction of pixels WAM marks as watermarked, inside
vs outside the PIE-Bench++ edit mask, after round-1 edits.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .data import EditStep
from .utils import get_device, pil_to_tensor, tensor_to_pil


def band_energy(res: np.ndarray, nbands: int = 8) -> np.ndarray:
    """Energy of a residual (H,W,3) in nbands radial frequency bands (luma-averaged over channels)."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(res.mean(2)))) ** 2
    h, w = f.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    edges = np.linspace(0, 1.0, nbands + 1)
    return np.array([f[(r >= a) & (r < b)].sum() for a, b in zip(edges[:-1], edges[1:])])


def residual_stats(orig: Image.Image, wm: Image.Image, t_orig: Image.Image, t_wm: Image.Image) -> dict:
    a = lambda im: np.asarray(im.convert("RGB").resize(orig.size), dtype=np.float32) / 255
    r0, rt = a(wm) - a(orig), a(t_wm) - a(t_orig)
    corr = float(np.corrcoef(r0.ravel(), rt.ravel())[0, 1])
    e0, et = band_energy(r0), band_energy(rt)
    # projection of the transformed residual onto the original residual, per band
    F0 = np.fft.fftshift(np.fft.fft2(r0.mean(2)))
    Ft = np.fft.fftshift(np.fft.fft2(rt.mean(2)))
    h, w = F0.shape
    yy, xx = np.mgrid[:h, :w]
    rad = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    edges = np.linspace(0, 1.0, len(e0) + 1)
    coh = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rad >= lo) & (rad < hi)
        num = np.real((Ft[m] * np.conj(F0[m])).sum())
        coh.append(float(num / max((np.abs(F0[m]) ** 2).sum(), 1e-12)))
    return {"res_corr": corr, "res_energy_ratio": float(et.sum() / max(e0.sum(), 1e-12)),
            **{f"band{i}_survival": c for i, c in enumerate(coh)},
            **{f"band{i}_energy0": float(v / e0.sum()) for i, v in enumerate(e0)}}


class VAERoundTrip:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None):
        from diffusers import AutoencoderKL

        self.device = device or get_device()
        dt = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dt,
                                                 variant="fp16" if dt == torch.float16 else None).to(self.device)

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> Image.Image:
        x = (pil_to_tensor(img, self.device) * 2 - 1).to(self.vae.dtype)
        z = self.vae.encode(x).latent_dist.mode()
        return tensor_to_pil((self.vae.decode(z).sample.float() + 1) / 2)


def identity_step(caption: str) -> EditStep:
    return EditStep(0, "identity", "keep the image unchanged", caption, caption)
