# PECCAVI-Edit: spotlight experiment plan

Target: CVPR 2027. **Enrolment (title/abstract/authors) 2026-11-10 AoE**, PDF 2026-11-16 AoE, supplementary 11-23. Results must be frozen by **Nov 9**; see issue #17. Each experiment maps to a paper table or
figure and to a GitHub issue (`research/issues/`). Status is honest: ✅ done, 🟡 code ready (needs the GPU host),
⬜ not started.

## The story in one line
A watermark doesn't fail because the edit is *large*. It fails because the edit *regenerates the carrier*.
So predict which carriers survive, spread a coded message across carriers that fail independently, and measure
robustness against the watermark support the edit actually destroyed.

## Preliminary evidence already in hand (stage 2, 90 images, n = 360 rows per watermark/editor)
**The invisible-edit paradox.** SDEdit (strength 0.5) changes the image *least* (median PSNR vs. original
18.0 → 14.6 dB over 4 rounds), yet removes **100%** of WAM, TrustMark and DWT-DCT-SVD detections after one edit.
FLUX.2-klein changes it *most* (13.2 → 7.6 dB) and keeps 58% of WAM detections after round 1.
`analysis/universality.py`: within an editor, pixel-change magnitude ranks survival well (in-editor AUC 0.76–0.98).
It transfers *ranking* to held-out editors (held-out AUC ≈ in-editor AUC) but not *calibration*: held-out log-loss
barely beats a per-editor base rate and is worse for SDEdit. So pixel change is the wrong variable, which motivates
`regeneration_map` (the watermark residual is resampled even where content is unchanged). Table:
`tables/stage2_universality.md`.

## Experiments

| # | Experiment | What makes it unique | Paper slot | Status | Issue |
|---|---|---|---|---|---|
| X1 | **Universal survival law**: R(ρ) for every method × editor against four EWO variants (pixel area, pixel energy, regeneration energy, GT mask) | Tests whether one curve in *regeneration*-EWO collapses all editors. If it does, editor identity stops mattering once you know what was regenerated | Fig. 1 (teaser) + Table 4 | 🟡 `run_ewo_analysis.py` | #1 |
| X2 | **Stability paradox**: correlate PECCAVI's NMP paraphrase stability with measured edit survival, per edit category | Direct empirical test of Eq. 3. Predicts a *negative* correlation for remove/replace edits | Fig. 3 | ⬜ needs the NMP extractor | #8 |
| X3 | **Survival atlas and co-failure structure**: survival matrix Z (regions × edits) from pilot signals; spectral clustering of the co-failure graph; Cantelli bound vs. empirical recovery | First measurement of *correlated* carrier erasure under editing; links coding theory to editing semantics | Fig. 4 + Sec. 2.7 | 🟡 `erasure.py` stats ready, needs the pilot-signal generator | #4 |
| X4 | **Main benchmark**: TPR@1e-3 over 8 edit categories, 5 editors, 7 methods | Standard, but on modern editors (FLUX.2-klein / Kontext) | Table 2 | ⬜ | #6 #7 #3 |
| X5 | **Held-out editors**: survival predictor trained on IP2P, MagicBrush, InstructDiffusion; tested on UltraEdit and Kontext | Generalisation claim | Table 3 | ⬜ | #5 |
| X6 | **Oracle gap**: carriers placed with the *true* edit mask (oracle) vs. the ESR predictor vs. NMP vs. random | Shows how close prediction gets to the best possible placement; few watermark papers report an oracle | Table 8 | ⬜ | #9 |
| X7 | **Watermark half-life**: rounds until detection < 50%, T ≤ 8, with exponential fits | One number per method/editor that summarises multi-turn robustness | Table 5 | ✅ T≤4 for 3 baselines (`stage2_decay.md`), ⬜ T=8 + ours | #10 |
| X8 | **ESR-hunting adaptive attacker**: attacker runs its own survival predictor (or saliency) and inpaints the top-k predicted carriers | Answers the reviewer question "what if the attacker knows where you embed?" | Table 10 / Sec. 4.10 | ⬜ | #11 |
| X9 | **Regime map: untrusted vs. cooperative editor**: re-embed oracle (≈100% detection) and guided editing, with the forgery result (guidance *writes* a mark on 5–12/18 clean images) | Positions the paper honestly. Cooperative settings are solved by re-embedding; the open problem is untrusted editors | Fig. 5 / Sec. 1 | ✅ data exists (`stage2`, `guided_test/forgery`) | #12 |
| X10 | **Fidelity and editability parity** (PSNR/SSIM/LPIPS before editing; CLIP-dir, edit success after) | Shows the robustness doesn't come from blocking the edit | Table 6 | 🟡 metrics exist | #3 |
| X11 | **Standard distortions + W-Bench regeneration** sanity table | Pre-empts the "only tested on your own attacks" critique | Supp. | ⬜ | #13 |
| X12 | **Redundancy frontier**: RS(K, 8) for K ∈ {8, 12, 16, 20, 24}: robustness vs. PSNR | Ablation | Table 9 | 🟡 `ErasureCode` ready | #3 |
| X13 | **Effective regeneration depth** τ(p): per-pixel diffusion-timestep equivalent of watermark destruction, and the content-vs-regeneration split of EWO (concepts adapted from few-step noise analysis, no method borrowed) | Editor-independent survival target for f_ψ | Fig. 1 panel + Sec. 2.2 | 🟡 proxy in `ewo.regeneration_map` | #18 |
| X14 | **Commercial black-box editors** (Firefly, GPT Image, Nano Banana): n=50 through real UIs | First evaluation of edit-robust watermarks on the editors people actually use | Supp. | ⬜ | #19 |

## Corrections already folded into the code (flag them in the paper too)
- **Payload**: RS(16, 8) over GF(2⁸) carries 8 info bytes = **48-bit message + 16-bit keyed tag** (`ErasureCode`),
  not "64-bit message + tag".
- **Co-failure (Eq. 23)**: replaced the joint-failure rate with **Cov(Zᵢ, Zⱼ)**, which is what Var[N_surv] (Eq. 44)
  needs. Only positive covariance is penalised in Eq. 27. `test_cofailure_cov_is_zero_for_independent_weak_carriers`
  shows why: two weak, independent carriers have covariance ≈ 0 but a large joint-failure rate.
- **False positives from the blind search**: the keyed tag must be counted. On random symbols, ≤ 3/300 decode to a
  validly tagged payload (`test_tag_rejects_wrong_key_and_random_symbols`); the real FPR must be measured on clean images (#3).

## Compute budget (L4 measured: IP2P 4.7 s/edit)
Main benchmark at 500 images × 8 categories × 5 editors × 5 methods ≈ 100k edits. At ~5 s average (more for Kontext)
that is about 140 GPU-h. One L4 cannot do it before 11-16; plan on 4–8 GPUs, or cut to 300 images and 3 editors
first (#15).
