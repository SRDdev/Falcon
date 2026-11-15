"""Edit-Watermark Overlap (PECCAVI-Edit Sec. 2.8): how much of the watermark's spatial support an edit
actually touched. Evaluation-only; the detector never sees these quantities. Numpy/scipy only.

Two variants are reported so that methods with global (VINE, TrustMark) and localised (WAM, ours)
support are compared fairly:
  rho_energy = sum_p A_w(p) M_e(p) / sum_p A_w(p)    energy-weighted, method-specific (Eq. 47)
  rho_area   = mean_p M_e(p)                          method-independent edit extent
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

DEFAULT_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9)


def _f(img) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def watermark_energy(original, watermarked) -> np.ndarray:
    """A_w(p) = ||x_w(p) - x(p)||^2 summed over channels (Eq. 46)."""
    d = _f(watermarked) - _f(original)
    return (d * d).sum(-1) if d.ndim == 3 else d * d


def soft_edit_map(before, after, sigma: float = 4.0, tau: float = 0.08) -> np.ndarray:
    """M_e in [0,1] for open-ended edits without a ground-truth mask: Gaussian-smoothed per-pixel colour
    change, saturating at tau. A pixel-space proxy; a feature-space map (DINO patch distance) is the
    planned replacement (see issues), and benchmarks with masks use the annotation instead."""
    d = np.abs(_f(after) - _f(before))
    d = d.mean(-1) if d.ndim == 3 else d
    return np.clip(gaussian_filter(d, sigma) / tau, 0.0, 1.0)


def regeneration_map(original, watermarked, edited, sigma: float = 8.0, hp_sigma: float = 1.5) -> np.ndarray:
    """M_e in [0,1] measured on the watermark signal rather than on content: 1 - local correlation between
    the embedded residual (x_w - x) and the high-passed residual left in the edit (x_edit - x). A pixel
    whose content is unchanged but whose watermark residual was resampled (SDEdit-style regeneration)
    scores ~1 here and ~0 in soft_edit_map."""
    def hp(a):
        a = a.mean(-1) if a.ndim == 3 else a
        return a - gaussian_filter(a, hp_sigma)

    x = _f(original)
    w, e = hp(_f(watermarked) - x), hp(_f(edited) - x)
    cov = gaussian_filter(w * e, sigma)
    var = np.sqrt(gaussian_filter(w * w, sigma) * gaussian_filter(e * e, sigma)) + 1e-12
    return np.clip(1.0 - cov / var, 0.0, 1.0)


def ewo_energy(energy: np.ndarray, edit_map: np.ndarray, eps: float = 1e-12) -> float:
    if energy.shape != edit_map.shape:
        raise ValueError(f"shape mismatch {energy.shape} vs {edit_map.shape}")
    return float((energy * edit_map).sum() / (energy.sum() + eps))


def ewo_area(edit_map: np.ndarray) -> float:
    return float(np.asarray(edit_map, dtype=float).mean())


def ewo_bin(rho, edges=DEFAULT_EDGES) -> np.ndarray:
    return np.digitize(np.asarray(rho, dtype=float), edges[1:-1])


def recovery_curve(rho, success, edges=DEFAULT_EDGES, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """R(rho) = P[success | rho in bin] with percentile-bootstrap 95% CIs (Eq. 51)."""
    rho, success = np.asarray(rho, dtype=float), np.asarray(success, dtype=float)
    b = ewo_bin(rho, edges)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(edges) - 1):
        x = success[b == i]
        lo = hi = np.nan
        if len(x) > 1:
            m = rng.choice(x, (n_boot, len(x))).mean(1)
            lo, hi = np.percentile(m, [2.5, 97.5])
        out.append({"bin": f"[{edges[i]:.2f},{min(edges[i + 1], 1):.2f})", "n": len(x),
                    "rate": float(x.mean()) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)
