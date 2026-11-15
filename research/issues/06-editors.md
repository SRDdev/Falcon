---
key: EDITORS
title: "Add editors: MagicBrush, InstructDiffusion, UltraEdit, FLUX.1 Kontext"
labels: infrastructure, gpu-host
milestone: "M1: infrastructure (Oct 31)"
---
## Goal
Extend `src/wmedit/editors.py` so the paper's five-editor suite runs: IP2P (have), MagicBrush, InstructDiffusion (seen), UltraEdit and FLUX.1 Kontext (held out). Keep FLUX.2-klein (have) as an extra modern editor.

## Tasks
- [ ] Each editor: deterministic seed through `stable_seed`, record peak memory and seconds per edit
- [ ] Check licences and gating (Kontext dev weights are gated and non-commercial; record in SETUP.md)
- [ ] Smoke test on 3 images and add to `tests/test_gpu_components.py`
- [ ] Kontext needs ≥ 24 GB in bf16; the L4 may need CPU offload. Measure it.

## Done when
`configs/main.yaml` lists all editors and a 5-image run completes.
