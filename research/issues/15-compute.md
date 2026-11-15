---
key: COMPUTE
title: "Compute plan: ≥4 GPUs for Nov 2–12, persistent storage for images"
labels: infrastructure, critical-path
milestone: "M1: infrastructure (Oct 31)"
---
## Problem
The main benchmark (≈100k edits) is about 140 GPU-h; Kontext and VINE are heavier. One L4 cannot finish before Nov 16. The GPU host also has **no persistent volume** (REPRODUCE.md), and the edited images are not in git, so EWO and region analyses must run where the images live.

## Tasks
- [ ] Provision 4–8 GPUs (A100/L40S preferred for Kontext/VINE)
- [ ] Persistent bucket for `experiments/*/images`; `scripts/sync.sh` push/pull to it
- [ ] Shard runs by editor; resumability already exists
- [ ] Fallback scope if compute slips: 300 images, 3 editors (IP2P, MagicBrush, Kontext), 4 baselines
