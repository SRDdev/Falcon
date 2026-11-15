---
key: EMBED
title: "Implement PECCAVI-Edit embedder/detector (multi-carrier Fourier templates + RS erasure decoding)"
labels: method, critical-path
milestone: "M1: infrastructure (Oct 31)"
---
## Goal
A `Watermark` subclass `PeccaviEdit` in `src/wmedit/watermarks.py` that embeds RS(16, 8) symbols (48-bit message + 16-bit keyed tag, `wmedit.erasure.ErasureCode`) into K selected carriers with keyed conjugate-symmetric Fourier templates (Eqs. 30–33), and detects blind with a windowed correlation search, soft erasures and RS decoding (Eqs. 35–40).

## Why it matters
Everything in Tables 2–9 needs it. The biggest open risk is **capacity**: can each region carry 8 bits at PSNR ≥ 38 dB after a VAE round-trip?

## Tasks
- [ ] Carriers: start from `erasure.grid_boxes` plus `select_carriers`; NMP/ESR scores come later (#8, #5)
- [ ] Fourier templates per (carrier, bit, channel) from key κ; strength α_j from Eq. 32
- [ ] Adaptive enhancement to a target SSIM (Eq. 33–34)
- [ ] Detector: windowed search, η_j reliability, erasure threshold τ_e, `ErasureCode.decode`
- [ ] **Day-1 capacity check**: per-symbol accuracy vs. PSNR on 50 PIE-Bench++ images, clean and after a VAE round-trip
- [ ] **Clean-image FPR** of the full search + tag on ≥ 1000 unwatermarked images (the search inflates false positives)
- [ ] Add to `REGISTRY`; GPU test in `tests/test_gpu_components.py`

## Done when
Clean bit accuracy ≥ 0.99 at PSNR ≥ 38 dB, measured FPR ≤ 1e-3, and it runs through `run_experiment.py`.

## Depends on
Nothing; `erasure.py` is ready. PECCAVI (CVPR 2026) code, if available, should be reused for the Fourier embedding.
