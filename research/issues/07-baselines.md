---
key: BASELINES
title: "Add baselines: VINE, Robust-Wide, PECCAVI (CVPR'26), Stable Signature, ZoDiac"
labels: infrastructure
milestone: "M1: infrastructure (Oct 31)"
---
## Goal
Wrap each baseline behind the `Watermark` interface in `src/wmedit/watermarks.py`.

## Notes
- **VINE** (Shilin-LU/VINE): NTU non-commercial, SDXL-Turbo based, heavy. It is the strongest editing baseline and essential.
- **Robust-Wide** (hurunyi/Robust-Wide): MIT, 64-bit; check the checkpoint is available.
- **PECCAVI**: the paper we extend; use the authors' code if it is ours, otherwise reimplement it from the paper.
- **Stable Signature**: generation-only; place it on the generated-image track (`configs/gentrack*`).
- **ZoDiac**: minutes per image; run on a 100-image subset and say so.
- WAM, TrustMark and DWT-DCT-SVD already exist.

## Done when
Each baseline passes the L0 round-trip and the clean-FPR test, and its licence is recorded in SETUP.md.
