"""Model-backed smoke tests (need weights; run on the GPU host: pytest -m gpu)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def image():
    from PIL import Image

    from wmedit.data import load_pie_bench
    from wmedit.utils import RESEARCH_ROOT

    s, imgs, _ = load_pie_bench(RESEARCH_ROOT / "datasets" / "hf_cache", ["6_change_attribute_color_40"], 1, 1, 2, 42)
    return s[0], imgs[s[0].id]


@pytest.mark.parametrize("name", ["wam", "trustmark", "dwtdct"])
def test_watermark_roundtrip_and_clean_image(name, image):
    from wmedit.watermarks import build_watermark

    _, img = image
    wm = build_watermark(name)
    bits = np.random.default_rng(1).integers(0, 2, wm.nbits).astype(np.uint8)
    w = wm.embed(img, bits)
    assert w.size == img.size
    assert wm.score(w, bits, 1e-3)["detected"]
    assert not wm.score(img, bits, 1e-3)["detected"]  # clean image must not be flagged


@pytest.mark.parametrize("name", ["wam", "trustmark"])
def test_differentiable_decoder_agrees(name, image):
    from wmedit.utils import pil_to_tensor
    from wmedit.watermarks import build_watermark

    _, img = image
    wm = build_watermark(name)
    bits = np.random.default_rng(2).integers(0, 2, wm.nbits).astype(np.uint8)
    w = wm.embed(img, bits)
    x = pil_to_tensor(w, wm.device).requires_grad_(True)
    logits = wm.bit_logits(x)
    soft = (logits[0] > 0).cpu().numpy().astype(np.uint8)
    assert (soft == bits).mean() >= 0.95
    logits.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_ip2p_loop_matches_official_pipeline(image):
    from wmedit.editors import InstructPix2Pix
    from wmedit.utils import pil_to_tensor

    s, img = image
    ed = InstructPix2Pix(steps=10)
    a, b = ed.edit(img, s.plan[0], seed=3), ed.reference_edit(img, s.plan[0], seed=3)
    diff = (pil_to_tensor(a) - pil_to_tensor(b)).abs().mean().item()
    assert diff < 2 / 255, diff


@pytest.mark.parametrize("name", ["treering", "gaussianshading"])
def test_generative_watermark_detects_own_generation_only(name):
    from wmedit.genwm import GEN_REGISTRY

    wm = GEN_REGISTRY[name]()
    bits = np.random.default_rng(3).integers(0, 2, wm.nbits).astype(np.uint8)
    cap, seed = "a photo of a red fox sitting in the snow", 11
    marked, clean = wm.generate(cap, bits, seed), wm.generate_clean(cap, seed)
    assert wm.score(marked, bits, 1e-3)["detected"]
    assert not wm.score(clean, bits, 1e-3)["detected"]
