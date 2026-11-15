---
key: PAPER
title: "Paper fixes: CVPR template, 8 pages, payload math, Eq. 23, citations, conclusion"
labels: writing, critical-path
milestone: "M3: enrol Nov 10, PDF Nov 16"
---
## Tasks
- [ ] Move to the CVPR 2027 template (currently "Anonymous ACL submission", 17 pages) and fit 8 pages; move Secs. 2.9–2.10 and Tables 8–9 to the supplement
- [ ] **Replace the Conclusion**: it is copied from the text paper ("AI-generated text", "linguistic coherence", "entropy-based masking")
- [ ] Payload: RS(16, 8) over GF(2⁸) = 48-bit message + 16-bit tag (matches `ErasureCode`)
- [ ] Eq. 23: co-failure = Cov(Z_i, Z_j), not the joint-failure rate; add argmax to Eq. 27
- [ ] Update PECCAVI to CVPR 2026 (Table 1 and refs); cite it in the third person for anonymity; keep spelling consistent
- [ ] Resolve `(?)`/`??`: InstructDiffusion, UltraEdit, FLUX.1 Kontext, Stable Signature, ZoDiac; fix the "at-a-glace" typo
- [ ] Add the invisible-edit paradox (stage 2) to the Intro as motivation
- [ ] Register the paper on OpenReview before the registration deadline (check the official date)
