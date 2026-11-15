---
key: REGION
title: "Run WAM region analysis (inside vs outside PIE-Bench++ edit mask)"
labels: experiment, gpu-host
milestone: "M1: infrastructure (Oct 31)"
---
## Goal
Run the existing `run_region_analysis.py` on stage 2: does WAM survive outside the edit mask, and does decoding from the unedited region alone recover payloads that whole-image decoding loses?

## Why it matters
This is the ground-truth-mask version of EWO and directly motivates spatially distributed carriers. The script was written on Oct 2 but has no results yet.

## Tasks
- [ ] `python run_region_analysis.py --config configs/stage2.yaml`
- [ ] Add `tables/stage2_region.md` via `analysis/analyze.py`, or a small formatter
- [ ] Report det_outside vs det_whole per editor and level

## Done when
`experiments/stage2/region/rows.csv` is committed and the table is in the report.

## Estimate
About 1 GPU-h (WAM detection only).
