"""CPU tests for erasure coding, carrier selection and Edit-Watermark Overlap (no torch needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("reedsolo")

from wmedit.erasure import (ErasureCode, bits_to_bytes, box_iou, bytes_to_bits, cantelli_failure_bound,  # noqa: E402
                            cofailure_cov, empirical_recovery, fuse_scores, grid_boxes, recovery_independent,
                            select_carriers, survival_moments)
from wmedit.ewo import (ewo_area, ewo_bin, ewo_energy, recovery_curve, regeneration_map, soft_edit_map,  # noqa: E402
                        watermark_energy)

KEY = b"secret"


def test_rs_recovers_up_to_k_minus_r_erasures():
    code = ErasureCode(16, 8, tag_bytes=2)
    msg = bytes(range(code.msg_bytes))
    sym = code.encode(msg, KEY)
    assert len(sym) == 16 and code.msg_bits == 48
    erased = [None if j in {0, 3, 5, 7, 9, 11, 13, 15} else s for j, s in enumerate(sym)]
    assert code.decode(erased, KEY) == msg
    assert code.decode([None] * 9 + sym[9:], KEY) is None  # 9 erasures > K - R


def test_rs_errors_and_erasures_bound():
    code = ErasureCode(16, 8)
    msg = b"\x01\x02\x03\x04\x05\x06"
    sym = code.encode(msg, KEY)
    noisy = list(sym)
    noisy[0], noisy[1] = None, None          # V = 2
    noisy[2] ^= 0xFF; noisy[3] ^= 0x0F; noisy[4] ^= 0x01  # E = 3 -> 2E + V = 8 = K - R
    assert code.decode(noisy, KEY) == msg


def test_tag_rejects_wrong_key_and_random_symbols():
    code = ErasureCode(16, 8)
    msg = b"abcdef"
    assert code.decode(code.encode(msg, KEY), b"other") is None
    rng = np.random.default_rng(0)
    hits = sum(code.decode(list(rng.integers(0, 256, 16)), KEY) is not None for _ in range(300))
    assert hits <= 3  # random symbols almost never decode to a validly tagged payload


def test_bits_roundtrip():
    bits = np.random.default_rng(1).integers(0, 2, 48).astype(np.uint8)
    assert np.array_equal(bytes_to_bits(bits_to_bytes(bits)), bits)


def test_cofailure_cov_is_zero_for_independent_weak_carriers():
    rng = np.random.default_rng(0)
    Z = (rng.random((2, 20000)) < 0.3).astype(int)  # weak, independent
    assert abs(cofailure_cov(Z)[0, 1]) < 0.01
    Zc = np.vstack([Z[0], Z[0]])                     # perfectly co-failing
    assert cofailure_cov(Zc)[0, 1] == pytest.approx(0.21, abs=0.01)


def test_selection_prefers_failure_diverse_carriers():
    # carriers 0,1,2 are the most robust but always fail together; 3,4 slightly weaker, independent
    rng = np.random.default_rng(0)
    shared = rng.random(4000) < 0.8
    Z = np.vstack([shared, shared, shared, rng.random(4000) < 0.75, rng.random(4000) < 0.75]).astype(int)
    s, cov = Z.mean(1), cofailure_cov(Z)
    top = select_carriers(s, 3, lam_c=0.0)
    div = select_carriers(s, 3, cov=cov, lam_c=10.0)
    assert set(top) == {0, 1, 2}
    assert {3, 4} <= set(div)
    assert empirical_recovery(Z[div], 2) > empirical_recovery(Z[top], 2)


def test_selection_overlap_penalty_and_limits():
    boxes = np.array([(0, 0, 10, 10), (0, 0, 10, 10), (20, 20, 30, 30)])
    iou = box_iou(boxes)
    assert iou[0, 1] == pytest.approx(1.0) and iou[0, 2] == 0
    assert select_carriers(np.array([1.0, 0.99, 0.5]), 2, iou=iou, lam_o=10) == [0, 2]
    with pytest.raises(ValueError):
        select_carriers(np.ones(2), 3)
    assert len(grid_boxes(512, 512, 4, 4)) == 16


def test_recovery_probability_and_cantelli():
    s = np.full(16, 0.7)
    p = recovery_independent(s, 8)
    from scipy.stats import binom
    assert p == pytest.approx(binom.sf(7, 16, 0.7))
    rng = np.random.default_rng(0)
    Z = (rng.random((16, 20000)) < 0.7).astype(int)
    mu, var = survival_moments(Z)
    assert 1 - empirical_recovery(Z, 8) <= cantelli_failure_bound(mu, var, 8) + 1e-9
    assert cantelli_failure_bound(5, 1, 8) == 1.0
    assert fuse_scores(np.array([1.0]), np.array([0.0]), 0.0)[0] < 0.01  # edit-removed carrier scores low


def test_ewo_energy_and_area():
    x = np.zeros((8, 8, 3))
    xw = x.copy()
    xw[:4] += 0.1                       # watermark energy only in the top half
    e = watermark_energy(x, xw)
    top = np.zeros((8, 8)); top[:4] = 1
    assert ewo_energy(e, top) == pytest.approx(1.0)
    assert ewo_energy(e, 1 - top) == pytest.approx(0.0)
    assert ewo_area(top) == 0.5
    assert list(ewo_bin([0.1, 0.3, 0.6, 0.99, 1.0])) == [0, 1, 2, 3, 3]


def test_soft_edit_map_localises_change():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[:, 32:] = 200
    m = soft_edit_map(a, b, sigma=1.0)
    assert m[:, 40:].mean() > 0.95 and m[:, :24].mean() < 0.05


def test_regeneration_map_sees_invisible_resampling():
    rng = np.random.default_rng(0)
    x = rng.random((96, 96, 3)) * 0.5 + 0.25
    wm = rng.normal(0, 0.02, (96, 96, 1))
    xw = x + wm
    kept = xw + rng.normal(0, 0.002, xw.shape)                 # watermark intact
    regen = xw.copy()
    regen[:, 48:] = x[:, 48:] + rng.normal(0, 0.02, (96, 48, 1))  # same content, residual resampled
    assert regeneration_map(x, xw, kept)[8:-8, 8:-8].mean() < 0.1
    m = regeneration_map(x, xw, regen)
    assert m[8:-8, 64:-8].mean() > 0.8 and m[8:-8, 8:32].mean() < 0.1
    assert soft_edit_map(x, regen)[:, 64:].mean() < 0.5         # pixel proxy underestimates it


def test_recovery_curve_bins():
    rho = np.array([0.1, 0.1, 0.6, 0.6, 0.9])
    ok = np.array([1, 1, 1, 0, 0])
    df = recovery_curve(rho, ok, n_boot=200)
    assert list(df.n) == [2, 0, 2, 1] and df.rate[0] == 1.0 and df.rate[2] == 0.5
