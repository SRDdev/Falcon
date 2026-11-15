---
key: MULTITURN
title: "[X7] Multi-turn to T=8 and watermark half-life"
labels: experiment
milestone: "M2: main results (Nov 7)"
---
## Goal
Extend sequential editing from 4 to 8 rounds and report the **half-life** (rounds until detection < 50%) per method × editor, from the existing exponential fits (`analysis/analyze.py` decay fits).

## Tasks
- [ ] `--set dataset.rounds=8` on the main config; `data.build_plan` already cycles categories
- [ ] Add a half-life column = ln 2 / exp_k, with bootstrap CIs over images
- [ ] Stage 2 baselines (T ≤ 4) are already in `tables/stage2_decay.md`

## Done when
Table 5 is filled for every method at T ∈ {1, 2, 4, 8}.
