# Experimental Setup

*Every quantity below is set in `configs/*.yaml`; exact software versions, GPU, and code hash for each run are in
`experiments/<name>/manifest_*.json`.*

## Data and edit protocol
**Real-image track.** PIE-Bench++ (UB-CVML-Group/PIE_Bench_pp, CC-BY-SA-4.0), categories 1–9 (change / add / delete
object, change content / pose / color / material, change background, change style); the mixed `0_random` category
is excluded so that each image's first edit has a single type. Images are 512×512. Stage 2 uses the first 10 images
of every category by id (90 images); the guided-editing test uses the 2nd–3rd image of every category (18 images),
so that the two images used to choose guidance settings are never evaluated.

**Generated-image track.** Tree-Ring and Gaussian Shading watermark the initial noise of a generation and cannot be
applied to photographs. For these, originals are SD-1.5 generations of the PIE-Bench++ source captions (5 per
category, 45 images), produced from the same base noise with and without the watermark.

**Sequential editing.** Level 0 is the watermarked image. Round 1 applies the image's own PIE-Bench++ edit
(InstructPix2Pix receives an instruction templated deterministically from the benchmark's `edit_action` annotation;
SDEdit receives the benchmark's target caption). Rounds 2–4 apply one image-agnostic edit each, in a fixed order of
increasing disruption — global attribute (night, winter, sunset, rain), background (beach, city street, forest,
desert), style (oil painting, anime, watercolor, cinematic) — drawn per image with a seeded RNG. Each round edits
the previous round's output. All prompts are stored in `experiments/<name>/prompts.jsonl`.

## Watermarks
| Watermark | Type | Payload | Detection test |
|---|---|---|---|
| WAM (MIT checkpoint) | post-hoc, neural, localising | 32 bits | one-sided binomial on matching bits, FPR 1e-3 (≥ 26/32) |
| TrustMark Q (ECC off) | post-hoc, neural | 100 bits | binomial, FPR 1e-3 (≥ 66/100) |
| DWT-DCT-SVD | post-hoc, classical | 32 bits | binomial, FPR 1e-3 (≥ 26/32) |
| Tree-Ring | generation-time, Fourier ring in z_T | zero-bit | noncentral-χ² p-value ≤ 1e-3 (official test) |
| Gaussian Shading | generation-time, keyed z_T sampling | 256 bits | binomial, FPR 1e-3 |

Payloads are drawn per (watermark, image) from a seeded RNG. The binomial threshold assumes fair-coin bits on
unwatermarked images; we verify this empirically by decoding every watermark on every image of the unwatermarked
track at every level.

## Editors
InstructPix2Pix (timbrooks/instruct-pix2pix, 50 Euler-ancestral steps, text guidance 7.5, image guidance 1.5 — the
released defaults) and SDEdit on SD-1.5 (noise strength 0.5, 50 steps, guidance 7.5). Larger editors (TurboEdit on
SDXL-Turbo, FLUX-class editors) did not fit the 12 GB GPU or required gated weights.

## Methods compared
- **B0** — editor on the unwatermarked image (edit-quality reference; FPR control).
- **B1 baseline** — editor on the watermarked image.
- **Re-embed** — decode the payload from the round's input, edit, embed the decoded payload into the output.
  Requires the watermark encoder at edit time.
- **Guided** — InstructPix2Pix whose sampling is steered by the watermark decoder's gradient (see method.md);
  payload decoded from the input; two pre-chosen strengths (λ = 0.005, 0.02; last 60% of steps).
- No legitimate existing watermark-preserving editing baseline could be run: SafeMark has no public code and targets
  domain-specific editors (DiffusionCLIP, Asyrp, Eff-Diff).

## Metrics and statistics
Watermark: detection rate (Wilson 95% CI), bit accuracy (bootstrap 95% CI), exact-payload rate, empirical FPR.
Edit: CLIP-T and its gain, CLIP directional similarity, CLIP-I, DINOv2 similarity, LPIPS, SSIM, PSNR, and PIE-Bench++
background preservation outside the edit mask in round 1. `edit_success` and `joint_success` follow the
pre-registered definitions in `analysis/PREREGISTRATION.md`. Method comparisons are paired by image: McNemar (exact)
for detection, Wilcoxon signed-rank for continuous metrics, Holm-corrected within each metric family.

## Failure-mode probes
On 27 images (3 per category): SD-1.5 VAE encode–decode only; InstructPix2Pix with a no-op instruction; the round-1
edit at image guidance 1.0/1.5/2.5/4.0 and at 10/25 steps; SDEdit with the source caption at strength 0.1–0.7.
For each, detection and the correlation between the watermark residual before and after the transform, per radial
frequency band.

## Compute
NVIDIA RTX 3060 12 GB (pilot, probes, guided and forgery tests) and NVIDIA L4 23 GB (stage 2, generated track), both
with torch 2.11.0+cu128. Measured: InstructPix2Pix 8.5 s/edit (3060) and 4.7 s/edit (L4); guided InstructPix2Pix
peak memory 5.5 GB.
