---
key: NMP
title: "[X2] Stability paradox: NMP paraphrase stability vs measured edit survival"
labels: experiment, spotlight
milestone: "M2: main results (Nov 7)"
---
## Goal
For each candidate region, compute PECCAVI's NMP paraphrase-stability score s_para and the measured edit survival s* from #4. Report their Spearman correlation per edit category.

## Why it matters
This tests Eq. 3 (paraphrase stability ⇏ edit stability) directly. The prediction is a positive correlation for style/global edits and a **negative** one for object removal/replacement, because the most stable object is the edit's target. That would be the paper's most quotable figure.

## Tasks
- [ ] NMP extractor (visual paraphrase: caption with KOSMOS-2, then img2img × 10, then saliency stability), reusing PECCAVI code where possible
- [ ] Join with Z per region; Spearman ρ with bootstrap CIs, per category
- [ ] Qualitative panel: NMP heat map, ESR heat map and edited image side by side

## Depends on
#4
