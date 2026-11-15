---
key: EWO
title: "[X1] Universal survival law: run EWO analysis on stage 2 and test curve collapse"
labels: experiment, spotlight, gpu-host
milestone: "M1: infrastructure (Oct 31)"
---
## Goal
Compute four Edit-Watermark Overlap variants for every stage-2 baseline edit and test whether recovery R(ρ) collapses onto one curve across editors when ρ is measured on the *regenerated watermark signal*.

## Why it matters
This is the teaser figure. Stage 2 already shows the **invisible-edit paradox**: SDEdit changes pixels least (18 dB PSNR) but kills 100% of detections, while FLUX.2-klein changes most and keeps 58% (`tables/stage2_universality.md`). If regeneration-EWO explains survival across editors where pixel change can't, that is the paper's central empirical law.

## Tasks
- [ ] On the GPU host (images on disk): `python run_ewo_analysis.py --config configs/stage2.yaml`
- [ ] Extend `analysis/universality.py` to use each EWO variant as the feature (held-out-editor AUC and log-loss)
- [ ] Figure: R(ρ) per editor, one panel per EWO variant, with 95% bootstrap CIs
- [ ] Sanity check: `regeneration_map` against the probes (VAE round-trip ≈ low ρ, SDEdit 0.5 ≈ high ρ)

## Done when
`experiments/stage2/ewo/{rows,curves}.csv` and `tables/stage2_ewo.md` exist, and the figure shows whether the regen-EWO curves overlap across editors (report either outcome).

## Depends on
Nothing; code is in `run_ewo_analysis.py` and `src/wmedit/ewo.py` (tested).

## Estimate
CPU-bound, about 1 h on the host.
