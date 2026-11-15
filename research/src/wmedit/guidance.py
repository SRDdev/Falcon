"""Watermark-guided editing (candidate proposed method; selected or rejected by the pilot evidence).

During InstructPix2Pix sampling, at selected late steps, decode the current x0-prediction to pixels,
compute the watermark decoder's loss against the payload, and move the latent down its gradient:

    x0_hat = z_t - sigma_t * eps            (eps fixed: d x0_hat / d z_t = I)
    L_wm   = BCE(decoder(VAE_dec(x0_hat)), payload)
    z_t   <- z_t - lambda * g / rms(g),     g = dL_wm / d x0_hat     (optionally masked)

Config keys (all optional):
  lam            step size in latent units (0 disables; must reproduce baseline exactly)
  start_frac     guide only steps i >= start_frac * steps (late, low-noise steps)
  every          guide every n-th eligible step
  payload        "decoded" (decode from the round input; realistic) | "oracle" (ground truth bits)
  keep_payload   decode once at round 1 and reuse in later rounds (default True)
  adaptive       if set {target: 0.9, max_mult: 4}: per-round multiplier on lam from the measured
                 bit agreement of the previous output (sequential retention mechanism, component E)
  final_pixel_steps  optional n steps of pixel-space refinement after decoding (decoder-only re-embed)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .methods import Method
from .utils import pil_to_tensor, tensor_to_pil


class GuidedEdit(Method):
    def __init__(self, cfg, editor, wm):
        super().__init__(cfg, editor, wm)
        if not getattr(editor, "name", "") == "ip2p":
            raise NotImplementedError("guidance implemented for InstructPix2Pix only")
        if wm is None or not wm.differentiable:
            raise ValueError(f"guidance needs a differentiable decoder, got {getattr(wm, 'name', None)}")
        self.vae = editor.pipe.vae
        self.lam = float(cfg.get("lam", 0.1))
        self.start_frac = float(cfg.get("start_frac", 0.5))
        self.every = int(cfg.get("every", 1))

    def _target(self, img, state):
        if self.cfg.get("payload", "decoded") == "oracle":
            return state["oracle_payload"]
        if "payload" not in state or not self.cfg.get("keep_payload", True):
            state["payload"] = self.wm.decode(img)
        return state["payload"]

    def run(self, img, step, seed, state):
        bits = torch.as_tensor(self._target(img, state), dtype=torch.float32, device=self.editor.device)[None]
        mult = state.get("lam_mult", 1.0)
        lam = self.lam * mult
        steps = self.editor.steps
        start = int(self.start_frac * steps)
        log: list[float] = []

        def hook(i, t, lat, x0, sigma):
            if lam == 0 or i < start or (i - start) % self.every:
                return lat
            with torch.enable_grad():
                x0 = x0.detach().float().requires_grad_(True)
                x = self.vae.decode((x0 / self.vae.config.scaling_factor).to(self.vae.dtype)).sample.float()
                logits = self.wm.bit_logits(((x + 1) / 2).clamp(0, 1))
                loss = F.binary_cross_entropy_with_logits(logits, bits)
                (g,) = torch.autograd.grad(loss, x0)
            log.append(float(loss))
            g = g / g.pow(2).mean().sqrt().clamp_min(1e-12)
            return (lat.float() - lam * g).to(lat.dtype)

        out = self.editor.edit(img, step, seed, hook=hook)
        acc = float((self.wm.decode(out) == bits[0].cpu().numpy()).mean())
        ad = self.cfg.get("adaptive")
        if ad:  # component E: raise guidance for the next round if agreement fell below target
            short = max(0.0, float(ad.get("target", 0.9)) - acc)
            state["lam_mult"] = float(min(ad.get("max_mult", 4.0), mult * (1 + ad.get("gain", 5.0) * short)))
        return out, {"lam": lam, "guided_steps": len(log), "wm_loss_first": log[0] if log else None,
                     "wm_loss_last": log[-1] if log else None, "agreement_out": acc}
