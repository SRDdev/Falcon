---
key: ORACLE
title: "[X6] Oracle-placement upper bound and carrier-selection ablation"
labels: experiment
milestone: "M2: main results (Nov 7)"
---
## Goal
Compare carrier placement policies at fixed K, code and strength: random, top-K NMP, top-K ESR, ESR + spatial diversity, ESR + co-failure, and an **oracle** that knows the true edit mask and avoids it.

## Why it matters
The oracle shows how much room is left for prediction to close. It turns Table 8 from an ablation into a gap analysis.

## Tasks
- [ ] Oracle uses the PIE-Bench++ round-1 mask (and the regeneration map for global edits)
- [ ] Report Avg / High-EWO / T=4 recovery and the fraction of the oracle gap closed
