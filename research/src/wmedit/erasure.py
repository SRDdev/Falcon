"""Carrier survival statistics, failure-diverse carrier selection and Reed-Solomon erasure coding
(PECCAVI-Edit Secs. 2.2-2.7). Numpy/scipy only so it runs and is tested without model weights.

Survival matrix convention: Z has shape (N carriers, M edits), Z[i, j] = 1 when carrier i is still
decodable after edit j (Eq. 18).
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
from reedsolo import ReedSolomonError, RSCodec


# ---------------------------------------------------------------- survival / co-failure statistics

def survival_rates(Z: np.ndarray) -> np.ndarray:
    """Empirical survival probability s_i (Eq. 19)."""
    return np.asarray(Z, dtype=float).mean(1)


def cofailure_cov(Z: np.ndarray) -> np.ndarray:
    """Cov(Z_i, Z_j) over edits. Equals the covariance of the failure indicators, and is the quantity
    that enters Var[N_surv] (Eq. 44). Unlike the joint-failure rate (1-z_i)(1-z_j), it is ~0 for two
    weak carriers that fail independently."""
    Z = np.asarray(Z, dtype=float)
    return np.cov(Z, bias=True) if Z.shape[1] > 1 else np.zeros((len(Z), len(Z)))


def cofailure_phi(Z: np.ndarray) -> np.ndarray:
    """Phi (Pearson) correlation of carrier survival; 0 where a carrier never or always survives."""
    c = cofailure_cov(Z)
    sd = np.sqrt(np.clip(np.diag(c), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = c / np.outer(sd, sd)
    return np.nan_to_num(phi)


def box_iou(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for boxes (N, 4) as (y0, x0, y1, x1)."""
    b = np.asarray(boxes, dtype=float)
    y0 = np.maximum(b[:, None, 0], b[None, :, 0])
    x0 = np.maximum(b[:, None, 1], b[None, :, 1])
    y1 = np.minimum(b[:, None, 2], b[None, :, 2])
    x1 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(y1 - y0, 0, None) * np.clip(x1 - x0, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area[:, None] + area[None, :] - inter + 1e-12)


def grid_boxes(h: int, w: int, rows: int, cols: int) -> np.ndarray:
    """Non-overlapping candidate regions R_i on a rows x cols grid."""
    ys, xs = np.linspace(0, h, rows + 1).astype(int), np.linspace(0, w, cols + 1).astype(int)
    return np.array([(ys[r], xs[c], ys[r + 1], xs[c + 1]) for r in range(rows) for c in range(cols)])


def fuse_scores(s_para: np.ndarray, s_edit: np.ndarray, lam_p: float = 0.5, eps: float = 1e-3) -> np.ndarray:
    """r_i = (s_para + eps)^lam_p * (s_edit + eps)^lam_e with lam_p + lam_e = 1 (Eq. 22)."""
    return (np.asarray(s_para) + eps) ** lam_p * (np.asarray(s_edit) + eps) ** (1 - lam_p)


def select_carriers(r: np.ndarray, k: int, cov: np.ndarray | None = None, iou: np.ndarray | None = None,
                    lam_c: float = 1.0, lam_o: float = 1.0, eps: float = 1e-3) -> list[int]:
    """Greedy maximiser of  sum log(r_i+eps) - lam_c sum_{i<j} cov_ij - lam_o sum_{i<j} IoU_ij  (Eq. 27,
    with the co-failure term as a covariance). Only positive covariance (shared failures) is penalised."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if k > n:
        raise ValueError(f"cannot select {k} carriers from {n}")
    cov = np.zeros((n, n)) if cov is None else np.clip(cov, 0, None)
    iou = np.zeros((n, n)) if iou is None else iou
    base = np.log(r + eps)
    chosen: list[int] = []
    for _ in range(k):
        gain = base - lam_c * cov[:, chosen].sum(1) - lam_o * iou[:, chosen].sum(1)
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
    return chosen


# ---------------------------------------------------------------- recovery probability (Sec. 2.7)

def recovery_independent(s: np.ndarray, need: int) -> float:
    """P[N_surv >= need] for independent carriers (Poisson-binomial, Eq. 43), by dynamic programming."""
    dist = np.zeros(len(s) + 1)
    dist[0] = 1.0
    for p in s:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[need:].sum())


def cantelli_failure_bound(mu: float, var: float, need: int) -> float:
    """Upper bound on P[N_surv < need] for mu > need (Eq. 45); 1.0 when the bound is vacuous."""
    if mu <= need:
        return 1.0
    return float(var / (var + (mu - need) ** 2))


def survival_moments(Z_sel: np.ndarray) -> tuple[float, float]:
    """(mu, sigma^2) of N_surv including covariance terms (Eq. 44)."""
    n = np.asarray(Z_sel, dtype=float).sum(0)
    return float(n.mean()), float(n.var())


def empirical_recovery(Z_sel: np.ndarray, need: int) -> float:
    """Fraction of edits after which at least `need` selected carriers survive."""
    return float((np.asarray(Z_sel).sum(0) >= need).mean())


# ---------------------------------------------------------------- authenticated RS payload (Secs. 2.4, 2.6)

def provenance_tag(message: bytes, key: bytes, nbytes: int = 2) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()[:nbytes]


class ErasureCode:
    """RS(K, R) over GF(2^8): R information symbols (message + keyed tag), K coded symbols, one per
    carrier. Recovers when 2E + V <= K - R (Eq. 40), E = symbol errors, V = erasures."""

    def __init__(self, k: int = 16, r: int = 8, tag_bytes: int = 2):
        if not 0 < r < k <= 255:
            raise ValueError("need 0 < R < K <= 255")
        if not 0 <= tag_bytes < r:
            raise ValueError("tag must leave room for the message")
        self.k, self.r, self.tag_bytes = k, r, tag_bytes
        self.msg_bytes = r - tag_bytes
        self._rs = RSCodec(k - r)

    @property
    def msg_bits(self) -> int:
        return 8 * self.msg_bytes

    def encode(self, message: bytes, key: bytes) -> list[int]:
        if len(message) != self.msg_bytes:
            raise ValueError(f"message must be {self.msg_bytes} bytes")
        return list(self._rs.encode(message + provenance_tag(message, key, self.tag_bytes)))

    def decode(self, symbols: list[int | None], key: bytes) -> bytes | None:
        """symbols[j] is None for an erased carrier. Returns the message, or None when decoding fails or
        the tag does not verify (treated as 'no watermark')."""
        if len(symbols) != self.k:
            raise ValueError(f"expected {self.k} symbols")
        erase = [j for j, s in enumerate(symbols) if s is None]
        if len(erase) > self.k - self.r:
            return None
        data = bytearray(0 if s is None else int(s) for s in symbols)
        try:
            payload = bytes(self._rs.decode(data, erase_pos=erase or None)[0])
        except ReedSolomonError:
            return None
        msg, tag = payload[:self.msg_bytes], payload[self.msg_bytes:]
        return msg if hmac.compare_digest(tag, provenance_tag(msg, key, self.tag_bytes)) else None


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
