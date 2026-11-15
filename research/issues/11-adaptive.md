---
key: ADAPTIVE
title: "[X8] ESR-hunting adaptive attacker and key-secrecy analysis"
labels: experiment, security
milestone: "M2: main results (Nov 7)"
---
## Goal
An attacker who knows the method but not the key: (a) trains its own survival predictor, or uses saliency, to guess the carriers; (b) inpaints the top-k guessed regions with SD-inpainting; (c) we measure detection vs. k and the image-quality cost of the attack.

## Why it matters
Reviewers will ask "what if the attacker targets the ESRs?" Report honestly where it breaks; the keyed random patching should hold up better.

## Tasks
- [ ] Guess-overlap metric: IoU between the attacker's top-k and the true carriers
- [ ] Detection and PSNR/LPIPS of the attacked image vs. k
- [ ] With and without keyed random patching
