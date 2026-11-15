"""Text-based image editors behind one interface: edit(img, step, seed, hook=None) -> PIL.

InstructPix2Pix is run through an explicit denoising loop (a line-for-line port of the diffusers
pipeline) so that guided variants can intervene at each step; `tests/test_editor_equivalence.py`
checks that with no hook it reproduces the official pipeline output.
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch
from PIL import Image

from .data import EditStep
from .utils import get_device, pil_to_tensor, tensor_to_pil

# hook(i, t, latents, x0_latent_pred, sigma) -> new latents
LatentHook = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class Editor(Protocol):
    name: str

    def edit(self, img: Image.Image, step: EditStep, seed: int, hook: LatentHook | None = None,
             **overrides) -> Image.Image: ...


def _fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    w8, h8 = w - w % 8, h - h % 8
    return img if (w8, h8) == (w, h) else img.resize((w8, h8), Image.LANCZOS)


class InstructPix2Pix:
    name = "ip2p"

    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", steps: int = 50,
                 text_cfg: float = 7.5, image_cfg: float = 1.5, device=None, dtype=torch.float16):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.device = device or get_device()
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.text_cfg, self.image_cfg = steps, text_cfg, image_cfg

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        # ordered [text, uncond, uncond] as in the diffusers pipeline
        return self.pipe._encode_prompt(prompt, self.device, 1, True, "")

    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None,
             steps: int | None = None, text_cfg: float | None = None, image_cfg: float | None = None):
        p, steps = self.pipe, steps or self.steps
        text_cfg = self.text_cfg if text_cfg is None else text_cfg
        image_cfg = self.image_cfg if image_cfg is None else image_cfg
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        with torch.no_grad():
            emb = self._encode_text(step.instruction)
            x = (pil_to_tensor(img, self.device) * 2 - 1).to(self.dtype)
            il = p.vae.encode(x).latent_dist.mode()
            il = torch.cat([il, il, torch.zeros_like(il)])
            p.scheduler.set_timesteps(steps, device=self.device)
            ts = p.scheduler.timesteps
            shape = (1, p.vae.config.latent_channels, img.height // 8, img.width // 8)
            lat = torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)
            lat = lat * p.scheduler.init_noise_sigma
        for i, t in enumerate(ts):
            with torch.no_grad():
                inp = p.scheduler.scale_model_input(torch.cat([lat] * 3), t)
                eps = p.unet(torch.cat([inp, il], 1), t, encoder_hidden_states=emb).sample
                e_txt, e_img, e_unc = eps.chunk(3)
                eps = e_unc + text_cfg * (e_txt - e_img) + image_cfg * (e_img - e_unc)
            if hook is not None:
                sigma = p.scheduler.sigmas[p.scheduler.step_index or i].to(lat)
                lat = hook(i, t, lat, (lat - sigma * eps).detach(), sigma)
            with torch.no_grad():
                lat = p.scheduler.step(eps, t, lat, generator=gen).prev_sample
        return self.decode(lat)

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> Image.Image:
        x = self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor).sample
        return tensor_to_pil((x.float() + 1) / 2)

    @torch.no_grad()
    def reference_edit(self, img, step: EditStep, seed: int) -> Image.Image:
        """Official diffusers pipeline; used only by the equivalence test."""
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(step.instruction, image=_fit(img), num_inference_steps=self.steps,
                         guidance_scale=self.text_cfg, image_guidance_scale=self.image_cfg,
                         generator=gen).images[0]


class SDEdit:
    """SDEdit (Meng et al., ICLR 2022) with SD-1.5: partial noising to `strength`, then denoise
    with the target caption."""
    name = "sdedit"

    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", steps: int = 50,
                 strength: float = 0.5, cfg: float = 7.5, device=None, dtype=torch.float16):
        from diffusers import StableDiffusionImg2ImgPipeline

        self.device = device or get_device()
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.strength, self.cfg = steps, strength, cfg

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None,
             strength: float | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided SDEdit not implemented")
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(step.target_caption, image=_fit(img), strength=strength or self.strength,
                         num_inference_steps=self.steps, guidance_scale=self.cfg,
                         generator=gen).images[0]


class Flux2Klein:
    """FLUX.2 [klein] 4B (Black Forest Labs, 2026; Apache-2.0): rectified-flow transformer with reference-image
    editing. Distilled: 4 steps, guidance 1.0 (model card). Takes the same instruction as InstructPix2Pix and runs
    at the input resolution (512x512) so that resizing never acts as an extra attack."""
    name = "flux2klein"

    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-4B", steps: int = 4, guidance: float = 1.0,
                 device=None, cpu_offload: bool = False):
        from diffusers import Flux2KleinPipeline

        self.device = device or get_device()
        dt = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dt)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.guidance = steps, guidance

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided FLUX.2 editing not implemented")
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(image=img, prompt=step.instruction, height=img.height, width=img.width,
                         num_inference_steps=self.steps, guidance_scale=self.guidance, generator=gen).images[0]


class QwenImageEdit:
    """Qwen-Image-Edit (Alibaba, 2025; Apache-2.0), 20B. VLM-conditioned editor: the input goes through Qwen2.5-VL for
    semantics and the VAE for appearance, so it is a different *mechanism* from instruction-trained (IP2P) or
    inversion-based editors and is held out from survival-predictor training. bf16 needs ~40 GB; use cpu_offload on
    smaller cards. Evaluation attack only."""
    name = "qwenedit"

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit", steps: int = 50, true_cfg: float = 4.0, device=None,
                 cpu_offload: bool = False):
        from diffusers import QwenImageEditPipeline

        self.device = device or get_device()
        dt = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dt)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.true_cfg = steps, true_cfg

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided Qwen-Image-Edit not implemented")
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(image=img, prompt=step.instruction, negative_prompt=" ", true_cfg_scale=self.true_cfg,
                        num_inference_steps=self.steps, generator=gen).images[0]
        return out.resize(img.size, Image.LANCZOS) if out.size != img.size else out


REGISTRY = {"ip2p": InstructPix2Pix, "sdedit": SDEdit, "flux2klein": Flux2Klein, "qwenedit": QwenImageEdit}


def build_editor(name: str, **kwargs) -> Editor:
    return REGISTRY[name](**kwargs)
