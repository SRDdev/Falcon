---
key: REGENDEPTH
title: "[X13] Effective regeneration depth: a PECCAVI survival variable inspired by few-step noise statistics"
labels: experiment, spotlight
milestone: "M2: main results (Nov 7)"
---
## Idea (concept only, no code from TurboEdit)
TurboEdit (arXiv:2408.00735) observed that inverted edit noises behave like noise from an *earlier* diffusion timestep, and that an edit splits into a content term (prompt change) and a trajectory term (re-sampling). PECCAVI-Edit adopts two principles from this, in its own terms:

1. **Effective regeneration depth** τ(p): for every pixel, the diffusion timestep whose noise level best explains how much of the embedded watermark residual was destroyed. This is a per-pixel scalar in diffusion units, independent of the editor. `regeneration_map` in `src/wmedit/ewo.py` is its first proxy; τ(p) is the calibrated version.
2. **Content vs regeneration decomposition of EWO**: ρ_content (what the prompt changed) vs ρ_regen (what was re-sampled). Our stage-2 "invisible-edit paradox" (SDEdit: lowest ρ_content, highest ρ_regen, 0% survival) is the motivating case.

## Why it is PECCAVI's
Survival prediction (Sec. 2.2) needs a target that transfers across editors. Edit magnitude does not (`tables/stage2_universality.md`). τ(p) is a candidate universal target: the survival predictor f_ψ learns to predict the regeneration depth an edit will impose on each carrier, and carriers are chosen where predicted depth is low and uncorrelated.

## Tasks
- [ ] Calibrate τ: noise the watermarked image to timestep t, denoise with the SD VAE/UNet at that t, measure residual correlation → monotone map corr → t; apply to `regeneration_map` output
- [ ] Add ρ_regen and ρ_content columns to `run_ewo_analysis.py` and the universality test
- [ ] Use τ as an auxiliary target in the survival predictor (#5) and report whether it improves held-out-editor AUC
- [ ] Few-step editors remain evaluation attacks only (FLUX.2-klein already in the suite; add others under #6)

## Non-goal
No TurboEdit method components (timestep shift, pseudo-guidance, noise clipping) are implemented or claimed.
