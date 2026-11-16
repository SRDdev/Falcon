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


def test_saliency_64x64_bucketed_partition():
    from wmedit.saliency import bucketed_tile_selection, dual_zone_tile_weights

    img = natural_like(4)
    sel = bucketed_tile_selection(img, grid=(8, 8), bucket_shape=(2, 2), fg_buckets=4)
    assert sel["boxes"].shape == (64, 4)
    assert sel["chosen_boxes"].shape == (16, 4)
    assert sel["zone_labels"].count("fg") == 4
    assert sel["zone_labels"].count("bg") == 12
    w_map, weights = dual_zone_tile_weights(img, grid=(8, 8), k_fg=4, k_bg=12)
    assert w_map.shape == (512, 512) and len(weights) == 64


def test_peccavi_hybrid_roundtrip_and_fg_bg_wipeout_survival():
    from wmedit.peccavi_edit import PeccaviHybrid
    from wmedit.watermarks import build_watermark

    wm = build_watermark("peccavi_hybrid", target_psnr=40.0)
    assert isinstance(wm, PeccaviHybrid)
    img = natural_like(5)
    msg = b"\xde\xad\xbe\xef\xca\xfe"
    w, boxes = wm.embed(img, msg, return_boxes=True)

    # Check PSNR >= 39.5 dB
    a, b = np.asarray(img, float), np.asarray(w, float)
    psnr = 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))
    assert psnr > 39.5

    # Clean round-trip & FPR checks
    s = wm.score(w, msg)
    assert s["detected"] and s["exact_payload"]
    assert wm.detect(img)["message"] is None
    assert PeccaviHybrid(key=b"wrong-key").detect(w)["message"] is None

    # Simulate foreground object replacement (destroy central 256x256 region)
    arr_fg = np.asarray(w).copy()
    rng = np.random.default_rng(7)
    arr_fg[128:384, 128:384] = rng.integers(0, 256, (256, 256, 3))
    det_fg = wm.detect(Image.fromarray(arr_fg.astype(np.uint8)))
    assert det_fg["message"] == msg

    # Simulate partial background wipeout (destroy 6 outer tiles)
    arr_bg = np.asarray(w).copy()
    for y0, x0, y1, x1 in boxes[:6]:
        arr_bg[y0:y1, x0:x1] = rng.integers(0, 256, (y1 - y0, x1 - x0, 3))
    det_bg = wm.detect(Image.fromarray(arr_bg.astype(np.uint8)))
    assert det_bg["message"] == msg

