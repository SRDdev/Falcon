# Related Work

*Sources and verification status: `literature/literature_table.csv`, `literature/gap_analysis.md`.*

**Post-hoc image watermarking.** Neural post-hoc watermarks embed a payload into an existing image and decode it with a
learned extractor. We study three that apply to arbitrary photographs and have reproducible, permissively licensed
implementations: Watermark Anything (WAM; Sander et al., ICLR 2025), which also localises the watermarked region;
TrustMark (Bui et al., 2023); and the classical DWT-DCT(-SVD) scheme from the `invisible-watermark` package used by the
Stable Diffusion reference code. VINE (Lu et al., ICLR 2025) is the strongest editing-robust post-hoc watermark we
found, but its non-commercial licence and SDXL-Turbo backbone put it outside our compute budget.

**Generation-time watermarks.** Tree-Ring (Wen et al., NeurIPS 2023), Gaussian Shading (Yang et al., CVPR 2024), RingID
and Stable Signature (Fernandez et al., ICCV 2023) watermark images *as they are generated*, through the initial noise
or a fine-tuned decoder, and cannot be applied to real photographs. ZoDiac (Zhang et al., NeurIPS 2024) carries a
Tree-Ring-style pattern into real images by per-image latent optimisation, at a cost of minutes per image.

**Watermark removal by regeneration and editing.** Diffusion regeneration provably removes pixel-level watermarks (Zhao et
al., NeurIPS 2024), and WAVES (ICML 2024) and W-Bench (VINE paper) benchmark removal under regeneration, global
editing (InstructPix2Pix, MagicBrush, UltraEdit) and local editing, reporting TPR at 0.1% FPR. All of these evaluate a
**single** transformation. Evennou & Kijak (IH&MMSec 2026) find that incidental removal varies strongly with the
editor and is lower for recent editors on local edits, so conclusions are editor-specific; ours are restricted to
InstructPix2Pix and SDEdit on SD-1.5. A group of recent unreviewed preprints (e.g. arXiv:2603.12949) argue
information-theoretically that diffusion editing erases watermarks; we cite them as background only.

**Robustness on the embedding side.** Robust-Wide (Hu et al., ECCV 2024), VINE-R, JigMark, RIW and EditGuard/OmniGuard
make the *watermark* survive editing by training or optimising against simulated edits. Robust-Wide also contains the
only sequential measurement we found: up to three InstructPix2Pix rounds on 30 images.

**Preservation on the editor side.** SafeMark (arXiv:2605.19511) fine-tunes domain-specific editors (DiffusionCLIP,
Asyrp, Eff-Diff) with a thresholded watermark-decoding loss, assuming the editing provider holds both the editor and
the watermark encoder–decoder. It evaluates single edits and does not compare against re-embedding the decoded payload,
which that same assumption permits. Guidance Watermarking (Gesny et al., ICLR 2026) steers diffusion *generation*
with gradients from a post-hoc decoder, without training. Our guided editor applies that mechanism to editing.

**Sequential and multi-turn editing.** I2EBench2.0, MagicBrush multi-turn sessions and related benchmarks evaluate
edit quality over several rounds, but not watermark retention.

**Position of this work.** We contribute (i) a paired, multi-round (L0–L4) retention benchmark across watermark types
and editors, with an explicitly calibrated detector and an unwatermarked control track; (ii) a probe-based analysis of
where in the editing pipeline the watermark is lost; and (iii) a direct test of whether in-loop guidance from the
decoder offers anything beyond decode → edit → re-embed.
