"""Tests for Consensus-Saliency WAM (CS-WAM).

Verifies:
1. Non-starving dual-zone JND spatial modulation strictly bounded in [0.90, 1.35].
2. Two-Stage Spatial Consensus Erasure Decoder correctly rejects edited outlier tiles
   under heavy simulated diffusion edit noise and recovers ground truth payload.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_dual_zone_tile_weights_bounds():
    from wmedit.saliency import dual_zone_tile_weights

    # Create synthetic test image
    rng = np.random.default_rng(42)
    img_arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)

    w_map, weights = dual_zone_tile_weights(
        img,
        grid=(8, 8),
        k_fg=8,
        k_bg=24,
        fg_weight=1.05,
        bg_weight=1.25,
        other_weight=0.95,
        extractor=None,  # Fallback to texture / edge saliency
    )

    assert w_map.shape == (512, 512)
    assert len(weights) == 64

    # Non-starving invariant: every pixel must be >= 0.90 and <= 1.35
    assert float(w_map.min()) >= 0.90 - 1e-5, f"Min weight {w_map.min()} violated non-starving floor 0.90"
    assert float(w_map.max()) <= 1.35 + 1e-5, f"Max weight {w_map.max()} violated JND ceiling 1.35"

    # Weight distribution check: 8 FG, 24 BG, 32 Other
    assert (weights == 1.25).sum() == 24
    assert (weights == 1.05).sum() == 8
    assert (weights == 0.95).sum() == 32


def test_cs_wam_registry_and_init(monkeypatch):
    from wmedit.watermarks import ConsensusSaliencyWAM, REGISTRY, WAM

    assert REGISTRY["cs_wam"] is ConsensusSaliencyWAM
    assert REGISTRY["wam_saliency"] is ConsensusSaliencyWAM

    # Mock WAM.__init__ to avoid requiring downloaded model weights on CPU
    monkeypatch.setattr(WAM, "__init__", lambda self, *args, **kwargs: None)

    # Verify instantiation with custom and legacy parameters
    cswam = ConsensusSaliencyWAM(
        logit_temp=2.5,
        top_k_tiles=12,
        trim_frac=0.1,
        spectral_blend=0.35,  # legacy kwarg ignored gracefully
    )
    assert cswam.logit_temp == 2.5
    assert cswam.top_k_tiles == 12
    assert cswam.trim_frac == 0.1
    assert cswam.name == "cs_wam"
    assert cswam.nbits == 32


def test_two_stage_consensus_rejects_diffusion_scrambled_tiles(monkeypatch):
    """Unit test for Two-Stage Spatial Consensus Erasure Decoder.

    Simulates a 512x512 image (64 tiles of 64x64):
    - Ground truth payload: 32 bits all ones (1, 1, ..., 1).
    - 24 tiles are unedited: weak true signal (logits ~ +1.5, soft vote ~ +0.65).
    - 40 tiles are heavily edited: random noise with extreme hallucinated logits (±8.0).
    Verifies that CS-WAM rejects the 40 outlier tiles via cosine consensus and decodes
    the true payload with 100% bit accuracy.
    """
    import torch
    from wmedit.watermarks import ConsensusSaliencyWAM, WAM

    # Mock WAM.__init__ to avoid requiring downloaded model weights on CPU
    monkeypatch.setattr(WAM, "__init__", lambda self, *args, **kwargs: None)

    cswam = ConsensusSaliencyWAM(logit_temp=2.0, top_k_tiles=16, trim_frac=0.125)

    gt_bits = np.ones(32, dtype=np.uint8)
    torch.manual_seed(42)

    # Construct synthetic bit predictions: [1, 32, 512, 512]
    # and synthetic mask: [1, 1, 512, 512]
    bitp = torch.zeros(1, 32, 512, 512)
    mask = torch.ones(1, 1, 512, 512) * 0.8

    from wmedit.erasure import grid_boxes
    boxes = grid_boxes(512, 512, 8, 8)

    # Tiles 0..23: unedited (true signal +1.5 with minor noise)
    for i in range(24):
        y0, x0, y1, x1 = boxes[i]
        bitp[0, :, y0:y1, x0:x1] = 1.5 + 0.3 * torch.randn(32, y1 - y0, x1 - x0)

    # Tiles 24..63: edited (extreme random logits ±8.0)
    for i in range(24, 64):
        y0, x0, y1, x1 = boxes[i]
        # Strong random sign and magnitude to simulate diffusion hallucination
        random_signs = torch.randint(0, 2, (32, 1, 1)).float() * 2 - 1
        bitp[0, :, y0:y1, x0:x1] = random_signs * (5.0 + 3.0 * torch.rand(32, y1 - y0, x1 - x0))

    # Mock detect_raw
    cswam.detect_raw = lambda img: (mask, bitp)

    dummy_img = Image.new("RGB", (512, 512))
    decoded_bits = cswam.decode(dummy_img)

    bit_acc = float((decoded_bits == gt_bits).mean())
    assert bit_acc == 1.0, f"CS-WAM decoded with bit accuracy {bit_acc:.2%}, expected 100%"
