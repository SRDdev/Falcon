"""CPU tests for the PECCAVI-Edit embedder/detector (numpy only)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("reedsolo")

from wmedit.peccavi_edit import PeccaviEdit, bandpass_template  # noqa: E402


def natural_like(seed=0, size=512):
    rng = np.random.default_rng(seed)
    b = np.stack([gaussian_filter(rng.standard_normal((size, size)), s) for s in (2, 6, 16)], -1)
    b = (b - b.min()) / (b.max() - b.min())
    return Image.fromarray((b * 255).astype(np.uint8))


def test_templates_are_keyed_orthogonal_unit_norm():
    rng = np.random.default_rng(0)
    t = np.stack([bandpass_template(np.random.default_rng(i), 128, 128, 3, 14) for i in range(8)])
    assert np.allclose(np.linalg.norm(t.reshape(8, -1), axis=1), 1, atol=1e-5)
    g = t.reshape(8, -1) @ t.reshape(8, -1).T
    assert np.abs(g - np.eye(8)).max() < 0.1
    assert abs(t.mean()) < 1e-6 and rng is not None


def test_roundtrip_fidelity_and_false_positives():
    wm = PeccaviEdit(target_psnr=40)
    img = natural_like(1)
    msg = b"\x12\x34\x56\x78\x9a\xbc"
    w = wm.embed(img, msg)
    a, b = np.asarray(img, float), np.asarray(w, float)
    assert 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2)) > 39.5
    s = wm.score(w, msg)
    assert s["detected"] and s["bit_acc"] > 0.97 and s["n_erased"] == 0
    assert wm.detect(img)["message"] is None                                   # clean image
    assert PeccaviEdit(key=b"other", target_psnr=40).detect(w)["message"] is None  # wrong key


def test_survives_carrier_erasure_up_to_code_limit():
    wm = PeccaviEdit(target_psnr=40)
    img = natural_like(2)
    msg = b"abcdef"
    w, boxes = wm.embed(img, msg, return_boxes=True)
    rng = np.random.default_rng(0)
    for n_kill, expect in ((8, True), (10, False)):
        arr = np.asarray(w).copy()
        for (y0, x0, y1, x1) in boxes[rng.choice(16, n_kill, replace=False)]:
            arr[y0:y1, x0:x1] = rng.integers(0, 256, arr[y0:y1, x0:x1].shape)
        det = wm.detect(Image.fromarray(arr.astype(np.uint8)))
        assert det["n_erased"] >= n_kill - 1
        assert (det["message"] == msg) is expect


def test_eta_separates_watermarked_from_clean():
    wm = PeccaviEdit(target_psnr=40)
    img = natural_like(3)
    w = wm.embed(img, b"zzzzzz")
    ew, ec = wm.detect(w)["eta"], wm.detect(img)["eta"]
    assert ew.mean() > 3 * ec.max() and ec.max() < 1.2  # tau_e (1.05) was calibrated on 54 PIE-Bench++ photos


def test_survival_scores_select_failure_diverse_carriers():
    wm = PeccaviEdit(target_psnr=40, grid=(6, 6))
    scores = np.linspace(0.2, 1.0, 36)
    boxes = wm.select(512, 512, scores)
    assert len(boxes) == 16 and len({tuple(b) for b in boxes}) == 16
