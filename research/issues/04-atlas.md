---
key: ATLAS
title: "[X3] Survival atlas: pilot-signal survival matrix Z and co-failure structure"
labels: experiment, spotlight
milestone: "M2: main results (Nov 7)"
---
## Goal
Generate the survival matrix Z (N candidate regions × M edits) with weak, mutually separable pilot signals (Eqs. 14–19), then analyse its co-failure structure.

## Why it matters
This is the first measurement of *correlated* carrier erasure under semantic editing. It produces the training targets for the survival predictor (#5) and the theory-vs-practice figure: the Cantelli bound (Eq. 45) against empirical recovery.

## Tasks
- [ ] Pilot embedder (one orthogonal template per grid region, strength below operational)
- [ ] M ≥ 24 edits per image (3 seen editors × 8 categories) on the predictor-training split
- [ ] `erasure.survival_rates`, `cofailure_cov`, `cofailure_phi`; spectral clustering of φ into "semantic failure modes"
- [ ] Plot the Cantelli bound vs. empirical P[N_surv < R] for top-K vs. diverse selection
- [ ] Ablate pilot strength (reviewer concern: weak pilots may misjudge survival)

## Done when
`experiments/atlas/Z.npz` exists, plus a figure showing that diverse selection lowers Var[N_surv] at equal mean survival.

## Estimate
About 300 images × 24 edits ≈ 7.2k edits ≈ 10 GPU-h on an L4.
