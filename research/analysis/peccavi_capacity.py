#!/usr/bin/env python
"""Go/no-go numbers for the PECCAVI-Edit embedder (#3), CPU only.

  python analysis/peccavi_capacity.py --images <dir with png/jpg> [--n 50] [--psnr 40]
  python analysis/peccavi_capacity.py --synthetic 30          # no data needed (filtered-noise images)

Reports: fidelity (PSNR/SSIM), clean recovery, per-symbol reliability eta on watermarked vs clean regions,
clean-image false positives (wrong key / no watermark), and robustness to JPEG, resize, blur, noise, crop-shift.
Writes tables/peccavi_capacity.md.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wmedit.peccavi_edit import PeccaviEdit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def synthetic(n: int, size: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    for i in range(n):
        base = np.stack([gaussian_filter(rng.standard_normal((size, size)), s) for s in (2, 6, 16)], -1)
        base = (base - base.min()) / (base.max() - base.min() + 1e-9)
        yield f"syn{i}", Image.fromarray((base * 255).astype(np.uint8))


def from_dir(d: Path, n: int, size: int = 512):
    for p in sorted(d.glob("*"))[:n]:
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            yield p.stem, Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); return Image.open(b).convert("RGB")


ATTACKS = {
    "none": lambda im: im,
    "jpeg90": lambda im: jpeg(im, 90), "jpeg70": lambda im: jpeg(im, 70), "jpeg50": lambda im: jpeg(im, 50),
    "resize0.75": lambda im: im.resize((int(im.width * .75), int(im.height * .75)), Image.BICUBIC).resize(im.size, Image.BICUBIC),
    "blur1": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
    "noise8": lambda im: Image.fromarray(np.clip(np.asarray(im).astype(float) + np.random.default_rng(0).normal(0, 8, np.asarray(im).shape), 0, 255).astype(np.uint8)),
    "shift3px": lambda im: Image.fromarray(np.roll(np.asarray(im), (3, -2), (0, 1))),
    "erase4regions": None,  # handled separately: regenerate 4 of 16 carriers with noise
}


def psnr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 10 * np.log10(255 ** 2 / (np.mean((a - b) ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--psnr", type=float, default=40.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    imgs = list(synthetic(a.synthetic)) if a.synthetic else list(from_dir(Path(a.images), a.n))
    wm = PeccaviEdit(target_psnr=a.psnr)
    wrong = PeccaviEdit(key=b"attacker-key", target_psnr=a.psnr)
    rng = np.random.default_rng(1)
    rows, fid, fpr = [], [], {"clean_no_wm": 0, "wrong_key": 0, "n": 0}
    for sid, img in imgs:
        msg = rng.bytes(wm.code.msg_bytes)
        w, boxes = wm.embed(img, msg, return_boxes=True)
        fid.append({"id": sid, "psnr": psnr(img, w)})
        # false positives: unwatermarked image, and watermarked image read with another key
        fpr["clean_no_wm"] += wm.detect(img)["message"] is not None
        fpr["wrong_key"] += wrong.detect(w)["message"] is not None
        fpr["n"] += 1
        eta_w, eta_c = wm.detect(w)["eta"], wm.detect(img)["eta"]
        for name, fn in ATTACKS.items():
            if fn is None:
                arr = np.asarray(w).copy()
                for (y0, x0, y1, x1) in boxes[rng.choice(len(boxes), 4, replace=False)]:
                    arr[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1] + rng.normal(0, 25, arr[y0:y1, x0:x1].shape), 0, 255)
                att = Image.fromarray(arr.astype(np.uint8))
            else:
                att = fn(w)
            s = wm.score(att, msg)
            rows.append({"id": sid, "attack": name, **{k: s[k] for k in ("bit_acc", "detected", "n_erased", "eta_mean")}})
        rows[-len(ATTACKS)]["eta_wm_mean"], rows[-len(ATTACKS)]["eta_clean_mean"] = eta_w.mean(), eta_c.mean()
    df, fd = pd.DataFrame(rows), pd.DataFrame(fid)
    summ = df.groupby("attack").agg(n=("id", "size"), bit_acc=("bit_acc", "mean"), detected=("detected", "mean"),
                                    erased=("n_erased", "mean"), eta=("eta_mean", "mean")).reindex(ATTACKS.keys())
    eta_w, eta_c = df.eta_wm_mean.dropna().mean(), df.eta_clean_mean.dropna().mean()
    out = [f"# PECCAVI-Edit capacity check {a.tag} (n={len(imgs)}, target PSNR {a.psnr} dB, K=16, RS(16,8), 48-bit msg)",
           f"\nFidelity: PSNR {fd.psnr.mean():.2f} ± {fd.psnr.std():.2f} dB",
           f"\nReliability eta: watermarked regions {eta_w:.3f}, clean regions {eta_c:.3f}, erasure threshold {wm.tau_e}",
           f"\nFalse positives: clean/no-watermark {fpr['clean_no_wm']}/{fpr['n']}, wrong key {fpr['wrong_key']}/{fpr['n']}\n",
           summ.to_markdown(floatfmt=".3f")]
    text = "\n".join(out)
    print(text)
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables" / f"peccavi_capacity{('_' + a.tag) if a.tag else ''}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
