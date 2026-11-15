# Pre-registered evaluation definitions

Written 2026-10-01, **before** any proposed-method result exists (only the pilot baseline was running).
Changing these after seeing method results would require reporting both versions.

## Objective A — did the requested edit happen? (per round r, input = L(r-1), output = L(r))
- `edit_success` := `clip_dir > 0` **and** `clip_t_gain > 0`
  (CLIP embedding moved toward the target caption, both directionally and in absolute alignment).
  A do-nothing editor scores clip_dir = 0, clip_t_gain = 0 and therefore fails.
- Continuous metrics are always reported alongside: clip_t, clip_t_gain, clip_dir, clip_i, dino, lpips, ssim, psnr,
  and for round 1 the PIE-Bench++ background-preservation metrics bg_psnr / bg_lpips.
- Edit quality of a method is judged primarily by the **paired difference to the same editor on the same
  image without any watermark-aware mechanism** (`baseline` on the same watermark track), and secondarily to
  B0 (editor on the unwatermarked image).

## Objective B — did the watermark survive? (per level L0..L4)
- Primary: `detected` := matching bits m satisfy P[Bin(k, 1/2) ≥ m] ≤ 1e-3 (k = 32 WAM, 100 TrustMark,
  32 DWT-DCT-SVD). Empirical FPR on the unwatermarked track is reported for every detector and level;
  if it exceeds 1e-3 materially, the threshold is recalibrated on clean images and both results are shown.
- Secondary: bit accuracy (mean, 95% CI), exact-payload recovery, p-value.
- Retention is reported (i) over all images and (ii) over images detected at L0.

## Joint
- `joint_success` := `edit_success` at round r **and** `detected` at level r. Reported per round and
  cumulatively (all rounds 1..r successful). No weighted composite score.

## Statistics
- Per-image paired design: identical images, prompts and editor seeds across watermarks and methods.
- Method vs. baseline: Wilcoxon signed-rank on paired per-image values (bit accuracy, clip_dir, lpips ...);
  McNemar test on paired binary outcomes (detected, edit_success). Holm correction across the
  (editor × watermark × level) family. 95% CIs by percentile bootstrap (10k resamples) over images.
