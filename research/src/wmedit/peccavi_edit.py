"""PECCAVI-Edit embedder / detector (Secs. 2.4-2.6), numpy-only so capacity, fidelity and false-positive numbers can
be produced without a GPU.

Payload: ErasureCode(K, R) -> K symbols of 8 bits, one symbol per carrier region.
Carrier j, bit l: keyed band-pass spread-spectrum template B_{j,l} (zero-mean, unit-norm, real) in the region's
luma channel; the symbol is embedded as  region += alpha_j * m(region) * sum_l (2 c_{j,l} - 1) B_{j,l},
where m is a texture mask (hides energy in textured areas) and alpha_j follows Eq. 32. Adaptive enhancement
(Eq. 33) rescales the whole residual to a target PSNR.
Detection is blind: for every carrier, correlate the high-passed luma of each candidate window with the templates
(Eq. 35; small translation search), take hard bits (Eq. 36), reliability eta_j (Eq. 37), erase below tau_e (Eq. 38),
Reed-Solomon error-and-erasure decode and verify the keyed tag (Eq. 39-40).

Detection decision for cross-method tables: a watermark is "detected" when the tag verifies (keyed, so random images
essentially never pass; measured empirically in tests/experiments). bit_acc is reported on the K*8 coded bits so the
method also appears in bit-level tables.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from .erasure import ErasureCode, box_iou, bytes_to_bits, grid_boxes, select_carriers

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _key_rng(key: bytes, *parts) -> np.random.Generator:
    h = hashlib.sha256(key + b"|" + "|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def bandpass_template(rng: np.random.Generator, h: int, w: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Real zero-mean unit-norm pattern whose spectrum is keyed noise restricted to the annulus f_lo..f_hi
    (cycles per region side). Equivalent to the conjugate-symmetric Fourier template of Eq. 31."""
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    r = np.sqrt(fy * fy + fx * fx)
    band = (r >= f_lo) & (r <= f_hi)
    spec = (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))) * band
    t = np.real(np.fft.ifft2(spec))
    t -= t.mean()
    return (t / (np.linalg.norm(t) + 1e-12)).astype(np.float32)


def highpass(y: np.ndarray, sigma: float) -> np.ndarray:
    return y - gaussian_filter(y, sigma)


def texture_mask(y: np.ndarray, size: int = 7, floor: float = 0.35) -> np.ndarray:
    """Local-contrast mask in [floor, 1]: more energy where the eye tolerates it."""
    mu = uniform_filter(y, size)
    var = np.clip(uniform_filter(y * y, size) - mu * mu, 0, None)
    s = np.sqrt(var)
    s = s / (np.percentile(s, 90) + 1e-6)
    return floor + (1 - floor) * np.clip(s, 0, 1)


class PeccaviEdit:
    """Watermark-interface compatible (embed/decode/score) without importing torch."""
    name = "peccavi_edit"
    differentiable = False

    def __init__(self, key: bytes = b"peccavi-edit-key", k: int = 16, r: int = 8, tag_bytes: int = 2,
                 grid: tuple[int, int] = (4, 4), band: tuple[float, float] = (6.0, 24.0), target_psnr: float = 40.0,
                 hp_sigma: float = 2.0, tau_e: float = 1.05, search: int = 4, search_step: int = 1,
                 strength_gamma: float = 1.0, alpha_range: tuple[float, float] = (0.5, 2.0), whiten: int = 15):
        self.key, self.code = key, ErasureCode(k, r, tag_bytes)
        self.k, self.grid, self.band, self.target_psnr = k, grid, band, target_psnr
        self.hp_sigma, self.tau_e, self.search, self.search_step = hp_sigma, tau_e, search, search_step
        self.gamma, self.alpha_range, self.whiten = strength_gamma, alpha_range, whiten
        self.nbits = self.code.msg_bits  # payload bits the pipeline draws per image (48 for RS(16,8), 2-byte tag)
        self.n_coded_bits = k * 8
        self._tpl: dict = {}

    # ------------------------------------------------------------------ carriers
    def candidate_boxes(self, h: int, w: int) -> np.ndarray:
        return grid_boxes(h, w, *self.grid)

    def select(self, h: int, w: int, scores: np.ndarray | None = None, cov: np.ndarray | None = None) -> np.ndarray:
        """Carrier boxes. With no survival scores every grid cell is a carrier (requires grid == K); with scores,
        the failure-diverse subset of Eq. 27."""
        boxes = self.candidate_boxes(h, w)
        if scores is None:
            if len(boxes) != self.k:
                raise ValueError("without survival scores the grid must have exactly K cells")
            return boxes
        idx = select_carriers(scores, self.k, cov=cov, iou=box_iou(boxes))
        return boxes[idx]

    def templates(self, j: int, h: int, w: int) -> np.ndarray:
        kk = (j, h, w)
        if kk not in self._tpl:
            self._tpl[kk] = np.stack([bandpass_template(_key_rng(self.key, j, l, h, w), h, w, *self.band)
                                      for l in range(8)])
        return self._tpl[kk]

    # ------------------------------------------------------------------ embed
    def embed(self, img: Image.Image, message: bytes | np.ndarray, scores: np.ndarray | None = None,
              cov: np.ndarray | None = None, return_boxes: bool = False):
        if not isinstance(message, (bytes, bytearray)):
            message = np.packbits(np.asarray(message, dtype=np.uint8)).tobytes()
        x = np.asarray(img.convert("RGB"), dtype=np.float32)
        y = x @ LUMA
        h, w = y.shape
        boxes = self.select(h, w, scores, cov)
        symbols = self.code.encode(bytes(message), self.key)
        r = np.ones(self.k) if scores is None else np.asarray(scores)[select_carriers(scores, self.k, cov, box_iou(self.candidate_boxes(h, w)))]
        alpha = np.clip(r ** self.gamma / np.mean(r ** self.gamma), *self.alpha_range)  # Eq. 32, relative
        resid = np.zeros_like(y)
        for j, (y0, x0, y1, x1) in enumerate(boxes):
            tpl = self.templates(j, y1 - y0, x1 - x0)
            bits = np.unpackbits(np.array([symbols[j]], dtype=np.uint8)).astype(np.float32)
            pat = ((2 * bits - 1)[:, None, None] * tpl).sum(0)
            resid[y0:y1, x0:x1] += alpha[j] * pat * texture_mask(y[y0:y1, x0:x1])
        # adaptive enhancement (Eq. 33): one global scale to hit the target PSNR on the RGB image
        mse_target = 255.0 ** 2 / 10 ** (self.target_psnr / 10)
        scale = np.sqrt(mse_target / (np.mean(resid ** 2) + 1e-12))
        out = np.clip(x + (scale * resid)[..., None], 0, 255)
        wm = Image.fromarray(out.round().astype(np.uint8))
        return (wm, boxes) if return_boxes else wm

    # ------------------------------------------------------------------ detect
    def _corr(self, yhp: np.ndarray, box, tpl: np.ndarray) -> np.ndarray:
        """Max-over-translations normalised correlation per bit (Eq. 35, translation search only)."""
        y0, x0, y1, x1 = box
        H, W = yhp.shape
        best = np.full(8, -np.inf)
        best_abs = -1.0
        offs = range(-self.search, self.search + 1, self.search_step) if self.search else [0]
        for dy in offs:
            for dx in offs:
                a, b, c, d = y0 + dy, y1 + dy, x0 + dx, x1 + dx
                if a < 0 or c < 0 or b > H or d > W:
                    continue
                win = yhp[a:b, c:d]
                win = win - win.mean()
                n = np.linalg.norm(win) + 1e-6
                cors = (tpl * win).sum((1, 2)) / n
                s = np.abs(cors).mean()
                if s > best_abs:
                    best_abs, best = s, cors
        return best

    def detect(self, img: Image.Image, boxes: np.ndarray | None = None) -> dict:
        x = np.asarray(img.convert("RGB"), dtype=np.float32)
        yhp = highpass(x @ LUMA, self.hp_sigma)
        if self.whiten:  # matched filter in coloured noise: divide by local high-pass energy
            yhp = yhp / (np.sqrt(np.clip(uniform_filter(yhp * yhp, self.whiten), 0, None)) + 1.0)
        h, w = yhp.shape
        boxes = self.select(h, w) if boxes is None else boxes
        symbols, eta, hard_bits = [], [], []
        for j, box in enumerate(boxes):
            tpl = self.templates(j, box[2] - box[0], box[3] - box[1])
            L = self._corr(yhp, box, tpl)
            bits = (L > 0).astype(np.uint8)
            hard_bits.append(bits)
            e = float(np.abs(L).mean()) * np.sqrt(tpl[0].size) / 4.0  # ~ SNR units; see tests for calibration
            eta.append(e)
            symbols.append(None if e < self.tau_e else int(np.packbits(bits)[0]))
        msg = self.code.decode(symbols, self.key)
        return {"message": msg, "symbols": symbols, "eta": np.array(eta), "coded_bits": np.concatenate(hard_bits),
                "n_erased": sum(s is None for s in symbols)}

    # ------------------------------------------------------------------ Watermark-interface glue
    def coded_bits(self, message: bytes) -> np.ndarray:
        return bytes_to_bits(bytes(self.code.encode(message, self.key)))

    def decode(self, img: Image.Image) -> np.ndarray:
        return self.detect(img)["coded_bits"]

    def score(self, img: Image.Image, message: bytes | np.ndarray, target_fpr: float = 1e-3) -> dict:
        if not isinstance(message, (bytes, bytearray)):
            message = np.packbits(np.asarray(message, dtype=np.uint8)).tobytes()
        det = self.detect(img)
        truth = self.coded_bits(bytes(message))
        matches = int((det["coded_bits"] == truth).sum())
        return {"bit_acc": matches / self.n_coded_bits, "matches": matches, "n_coded_bits": self.n_coded_bits, "detected": det["message"] == bytes(message),
                "tag_verified": det["message"] is not None, "n_erased": det["n_erased"],
                "exact_payload": det["message"] == bytes(message), "eta_mean": float(det["eta"].mean())}
