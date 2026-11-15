"""Image / edit metrics. All operate on PIL images; models are loaded once and kept on device.

Metric                 What it measures                                    Why / limitation
clip_t                 CLIP(img) . CLIP(target caption)                    edit happened at all; CLIP is
                                                                           insensitive to small/local edits
clip_t_gain            clip_t(out) - clip_t(in)                            controls for captions the input
                                                                           already satisfies
clip_dir               cos(dImg, dText), d = CLIP(out)-CLIP(in),           direction of change matches the
                       target-caption minus source-caption                 requested change; noisy for tiny edits
clip_i, dino           cos sim of image embeddings (out vs in)             content/structure preservation;
                                                                           DINOv2 is more structure-sensitive
lpips, ssim, psnr      perceptual / structural / pixel distance            preservation; reward "doing nothing",
                       (out vs in)                                         so never read alone
bg_psnr, bg_lpips      same, restricted to outside the PIE-Bench++ mask    preservation of non-edited region
                       (round 1 only)                                      (PIE-Bench protocol)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import get_device, pil_to_tensor


class ImageMetrics:
    def __init__(self, clip_model: str = "ViT-L-14", clip_pretrained: str = "openai",
                 dino_model: str = "facebook/dinov2-small", device=None):
        import lpips
        import open_clip
        from transformers import AutoModel

        self.device = device or get_device()
        self.clip, _, self.clip_pre = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
        self.clip = self.clip.to(self.device).eval()
        self.tok = open_clip.get_tokenizer(clip_model)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        self.dino = AutoModel.from_pretrained(dino_model).to(self.device).eval()
        self._txt_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def clip_img(self, img: Image.Image) -> torch.Tensor:
        e = self.clip.encode_image(self.clip_pre(img).unsqueeze(0).to(self.device))
        return F.normalize(e.float(), dim=-1)[0]

    @torch.no_grad()
    def clip_txt(self, text: str) -> torch.Tensor:
        if text not in self._txt_cache:
            e = self.clip.encode_text(self.tok([text]).to(self.device))
            self._txt_cache[text] = F.normalize(e.float(), dim=-1)[0]
        return self._txt_cache[text]

    @torch.no_grad()
    def dino_emb(self, img: Image.Image) -> torch.Tensor:
        x = pil_to_tensor(img.resize((224, 224), Image.BICUBIC), self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        out = self.dino(pixel_values=(x - mean) / std)
        return F.normalize(out.pooler_output.float(), dim=-1)[0]

    @torch.no_grad()
    def lpips_dist(self, a: Image.Image, b: Image.Image, mask: np.ndarray | None = None) -> float:
        ta, tb = (pil_to_tensor(x, self.device) * 2 - 1 for x in (a, b))
        if mask is None:
            return float(self.lpips(ta, tb))
        m = torch.from_numpy(mask).float().to(self.device)[None, None]
        lp = lpips_spatial(self.lpips, ta, tb)  # spatial map, averaged over the mask below
        lp = F.interpolate(lp, size=m.shape[-2:], mode="bilinear", align_corners=False)
        return float((lp * m).sum() / m.sum().clamp_min(1))

    def pair(self, inp: Image.Image, out: Image.Image, source_caption: str, target_caption: str,
             keep_mask: np.ndarray | None = None) -> dict:
        if inp.size != out.size:
            out = out.resize(inp.size, Image.LANCZOS)
        ci, co = self.clip_img(inp), self.clip_img(out)
        ts, tt = self.clip_txt(source_caption), self.clip_txt(target_caption)
        di, dt = co - ci, tt - ts
        a, b = np.asarray(inp), np.asarray(out)
        res = {
            "clip_t": float(co @ tt),
            "clip_t_in": float(ci @ tt),
            "clip_t_gain": float(co @ tt - ci @ tt),
            "clip_dir": float(F.cosine_similarity(di, dt, dim=0)) if di.norm() > 1e-6 else 0.0,
            "clip_i": float(co @ ci),
            "dino": float(self.dino_emb(out) @ self.dino_emb(inp)),
            "lpips": self.lpips_dist(inp, out),
            "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
            "psnr": float(peak_signal_noise_ratio(a, b, data_range=255)),
        }
        if keep_mask is not None and 0.02 < keep_mask.mean() < 0.98:
            m = keep_mask.astype(bool)
            res["bg_psnr"] = float(peak_signal_noise_ratio(a[m], b[m], data_range=255))
            res["bg_lpips"] = self.lpips_dist(inp, out, keep_mask.astype(np.float32))
        return res


def lpips_spatial(model, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spatial LPIPS map (1,1,H',W') using the library's spatial mode."""
    model.spatial = True
    try:
        return model(a, b)
    finally:
        model.spatial = False


def image_quality(ref: Image.Image, img: Image.Image) -> dict:
    """Imperceptibility of the watermark itself (watermarked vs original)."""
    a, b = np.asarray(ref), np.asarray(img.resize(ref.size))
    return {"psnr": float(peak_signal_noise_ratio(a, b, data_range=255)),
            "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255))}
