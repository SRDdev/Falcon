---
key: PREDICTOR
title: "[X5] Train Edit-Survival Predictor f_ψ and pairwise co-failure head g_ψ"
labels: method
milestone: "M2: main results (Nov 7)"
---
## Goal
Train f_ψ(h_i, u_i, φ(x)) → ŝ_edit and g_ψ(h_i, h_j, u_i − u_j) → ĉ_ij (Eqs. 20–26) on #4 targets with a frozen DINOv2 backbone.

## Tasks
- [ ] Dataset split: predictor-train images must not overlap the evaluation set
- [ ] Loss: L_surv (soft BCE) + λ_co L_co (MSE on covariance, not joint failure)
- [ ] Calibration plot and AUC on held-out images, and separately on held-out editors (UltraEdit, Kontext)
- [ ] Plug into `PeccaviEdit` carrier selection via `fuse_scores` + `select_carriers`

## Done when
Held-out-editor survival AUC is reported and the predictor is used by the embedder.

## Depends on
#4, #6
