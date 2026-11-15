---
key: REGIME
title: "[X9] Regime map: untrusted vs cooperative editor (re-embed oracle, guided editing, forgery)"
labels: experiment, writing
milestone: "M3: enrol Nov 10, PDF Nov 16"
---
## Goal
Turn existing results into one figure and one paragraph that position the paper.

## Existing evidence
- Re-embedding keeps ≈100% detection through 4 rounds on all 3 editors (`tables/stage2_sequential_det_rate.md`)
- Decoder-guided editing: L4 detection 0.72–0.89 on 18 held-out images (`tables/guided_test_sequential_det_rate.md`)
- Forgery: guidance *creates* detectable marks on 5–12/18 clean images; forced re-embed forges 18/18 (`experiments/guided_test/forgery/rows.csv`)

## Tasks
- [ ] Figure: x = editor access (none → decoder → encoder), y = L4 detection, plus a forgery-rate inset
- [ ] Paragraph for Sec. 1: cooperative editing is solved by re-embedding; PECCAVI-Edit targets the untrusted editor
