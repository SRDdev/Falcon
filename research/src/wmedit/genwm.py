"""Generation-time watermarks (cannot be applied to real photos): Tree-Ring and Gaussian Shading.

Both watermark the initial noise z_T of a Stable Diffusion generation and are detected by DDIM-inverting
the image back to z_T with an empty prompt (guidance 1), as in the official code. We use SD-1.5 for both
generation and inversion (the papers used SD-2.1-base; neither method depends on the model).

Tree-Ring   ported from github.com/YuxinWenRick/tree-ring-watermark @3015283 (MIT): `ring` pattern,
            radius 10, channel 3, complex injection; zero-bit test = the paper's noncentral-chi2 p-value.
Gaussian    ported from github.com/bsmhmmlf/Gaussian-Shading @09c678f (MIT), non-ChaCha variant
Shading     (key XOR, truncated-Gaussian sampling), ch=1, hw=8 -> 256-bit payload; bit test as other wms.
The ports keep the official math; they only replace hard-coded .cuda() and unseeded RNG with seeded,
device-agnostic code so payloads/keys can be regenerated at evaluation time.
"""
from __future__ import annotations

import numpy as np
import scipy.stats
import torch
from PIL import Image
from scipy.stats import norm, truncnorm

from .utils import get_device, pil_to_tensor, stable_seed, tensor_to_pil
from .watermarks import Watermark

_SD: dict = {}


def sd15(device=None):
    """One shared SD-1.5 pipeline (generation) + DDIM inverse scheduler (detection)."""
    key = str(device or get_device())
    if key not in _SD:
        from diffusers import DDIMInverseScheduler, DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

        dev = torch.device(key)
        dt = torch.float16 if dev.type == "cuda" else torch.float32
        mid = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(mid, torch_dtype=dt, variant="fp16" if dt == torch.float16 else None,
                                                       safety_checker=None, requires_safety_checker=False).to(dev)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # as Tree-Ring
        pipe.set_progress_bar_config(disable=True)
        inv = DDIMInverseScheduler.from_config(DDIMScheduler.from_config(pipe.scheduler.config).config)
        _SD[key] = (pipe, inv)
    return _SD[key]


class _Generative(Watermark):
    generative = True
    steps, inv_steps, cfg = 50, 50, 7.5

    def __init__(self, device=None, **_):
        self.device = device or get_device()
        self.pipe, self.inv = sd15(self.device)
        self._empty = None

    # ---- generation
    @torch.no_grad()
    def _gen(self, caption: str, z: torch.Tensor) -> Image.Image:
        return self.pipe(caption, latents=z.to(self.pipe.unet.dtype), num_inference_steps=self.steps,
                         guidance_scale=self.cfg).images[0]

    def base_noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator("cpu").manual_seed(seed)
        return torch.randn(1, 4, 64, 64, generator=g).to(self.device)

    def generate_clean(self, caption: str, seed: int) -> Image.Image:
        return self._gen(caption, self.base_noise(seed))

    # ---- detection: DDIM inversion with empty prompt, guidance 1
    @torch.no_grad()
    def invert(self, img: Image.Image) -> torch.Tensor:
        p = self.pipe
        if self._empty is None:
            self._empty = p.encode_prompt("", self.device, 1, False)[0]
        x = (pil_to_tensor(img.resize((512, 512), Image.BICUBIC), self.device) * 2 - 1).to(p.vae.dtype)
        lat = p.vae.encode(x).latent_dist.mode() * p.vae.config.scaling_factor
        self.inv.set_timesteps(self.inv_steps, device=self.device)
        for t in self.inv.timesteps:
            eps = p.unet(lat, t, encoder_hidden_states=self._empty).sample
            lat = self.inv.step(eps, t, lat).prev_sample
        return lat.float()


class TreeRing(_Generative):
    name = "treering"
    nbits = 0  # zero-bit
    w_channel, w_radius, w_seed = 3, 10, 999999

    def __init__(self, **kw):
        super().__init__(**kw)
        size = 64
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        circ = lambda r: torch.tensor(((x - size // 2) ** 2 + (y - size // 2) ** 2) <= r ** 2)  # circle_mask
        self.mask = torch.zeros(1, 4, size, size, dtype=torch.bool)
        self.mask[:, self.w_channel] = circ(self.w_radius)
        # get_watermarking_pattern(..., w_pattern='ring')
        g = torch.Generator("cpu").manual_seed(self.w_seed)
        gt_init = torch.randn(1, 4, size, size, generator=g)
        patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        tmp = patch.clone()
        for i in range(self.w_radius, 0, -1):
            m = circ(i)
            for j in range(patch.shape[1]):
                patch[:, j, m] = tmp[0, j, 0, i].item()
        self.mask, self.patch = self.mask.to(self.device), patch.to(self.device)

    def watermarked_noise(self, seed: int) -> torch.Tensor:  # inject_watermark(..., 'complex')
        z = self.base_noise(seed)
        f = torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2))
        f[self.mask] = self.patch[self.mask].clone()
        return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-1, -2))).real

    def generate(self, caption: str, bits, seed: int) -> Image.Image:
        return self._gen(caption, self.watermarked_noise(seed))

    def score(self, img, bits, target_fpr):
        z = self.invert(img)
        f = torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2))[self.mask].flatten()
        tgt = self.patch[self.mask].flatten()
        tgt = torch.cat([tgt.real, tgt.imag])
        f = torch.cat([f.real, f.imag])
        sigma = f.std()
        lam = (tgt ** 2 / sigma ** 2).sum().item()
        xw = (((f - tgt) / sigma) ** 2).sum().item()
        p = float(scipy.stats.ncx2.cdf(x=xw, df=len(tgt), nc=lam))  # get_p_value
        l1 = float(torch.abs(torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2))[self.mask] - self.patch[self.mask]).mean())
        return {"bit_acc": np.nan, "matches": np.nan, "p_value": p, "detected": bool(p <= target_fpr),
                "threshold_matches": np.nan, "exact_payload": np.nan, "tr_l1": l1}


class GaussianShading(_Generative):
    name = "gaussianshading"
    ch, hw = 1, 8
    nbits = 4 * 8 * 8 // ch  # 256

    key_seed = 20241  # one secret key per deployment (as a service would hold); payload varies per image

    def _key(self) -> torch.Tensor:
        g = torch.Generator("cpu").manual_seed(self.key_seed)
        return torch.randint(0, 2, (1, 4, 64, 64), generator=g)

    def _wm_tensor(self, bits) -> torch.Tensor:
        return torch.as_tensor(np.asarray(bits), dtype=torch.long).reshape(1, 4 // self.ch, 64 // self.hw, 64 // self.hw)

    def watermarked_noise(self, bits, seed: int) -> torch.Tensor:  # create_watermark_and_return_w + truncSampling
        sd = self._wm_tensor(bits).repeat(1, self.ch, self.hw, self.hw)
        m = ((sd + self._key()) % 2).flatten().numpy()
        rng = np.random.default_rng(stable_seed("gs_sample", seed))
        ppf = [norm.ppf(j / 2.0) for j in range(3)]
        lo = np.array([ppf[0], ppf[1]])[m]
        hi = np.array([ppf[1], ppf[2]])[m]
        z = truncnorm.rvs(lo, hi, random_state=rng)
        return torch.from_numpy(z).float().reshape(1, 4, 64, 64).to(self.device)

    def generate(self, caption: str, bits, seed: int) -> Image.Image:
        return self._gen(caption, self.watermarked_noise(bits, seed))

    def decode(self, img) -> np.ndarray:  # diffusion_inverse + eval_watermark
        z = self.invert(img)
        rm = (z.cpu() > 0).long()
        rsd = (rm + self._key()) % 2
        thr = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        parts = torch.cat(torch.split(rsd, 4 // self.ch, dim=1), 0)
        parts = torch.cat(torch.split(parts, 64 // self.hw, dim=2), 0)
        parts = torch.cat(torch.split(parts, 64 // self.hw, dim=3), 0)
        vote = (parts.sum(0) > thr).long()
        return vote.flatten().numpy().astype(np.uint8)


GEN_REGISTRY = {"treering": TreeRing, "gaussianshading": GaussianShading}
