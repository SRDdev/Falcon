# Gap Analysis: Watermark Retention Under Text-Based Image Editing

Audit date: 2026-10-01. Sources were checked against the arXiv API and abstract/HTML pages, GitHub API and READMEs, and the ICLR 2026 virtual site. "From knowledge" means a venue I did not re-check during this audit. UNVERIFIED means I could not confirm it. The full table is in `literature_table.csv`.

## (a) Verdicts on claims from the prior notes

| Claim | Verdict |
|---|---|
| **SafeMark, arXiv:2605.19511** | **Exists. Mostly correct.** The full title is "Are Watermarked Images Editable? SafeMark for Watermark-Preserving Text-Guided Image Editing" (Wu, Li, Li, Zhang, Liu, Ni), submitted 2026-05-19, arXiv only with no venue. It fine-tunes the editor with a *thresholded watermark-decoding loss*, so the claim is correct. **Caveats the notes left out:** (1) the editors are DiffusionCLIP, Asyrp and Eff-Diff, which are domain-specific editors that need fine-tuning. They are *not* InstructPix2Pix, MagicBrush or TurboEdit. (2) The watermarks are HiDDeN, VINE, Stable Signature and SleeperMark. (3) It tests single edits only, with **no sequential editing**. (4) The data is LSUN/CelebA/AFHQ at 100 images per set. "No public code" is **confirmed**: I found no repository, and the paper mentions only a PyTorch implementation. |
| **"Editing Away the Evidence", arXiv:2603.12949** | **The paper exists. The venue claim is FALSE.** The arXiv record (2026-03-13, eess.IV) shows a preprint with no venue, so a CVPR 2025 workshop is impossible. The content claim is roughly right: theory plus experiments on StegaStamp, TrustMark and VINE under InstructPix2Pix/UltraEdit, drag-based and composition editors, with bit accuracy around 53-61% at strong edits. **Credibility warning:** it shares the co-authors "Emily Davis" and "Finn Carter" with a group of near-identical "theoretical and empirical" papers: 2510.05978, 2511.05598, 2603.04696 and 2603.29736. The related 2511.10933 and 2602.20680 use the same framing. None of these is peer-reviewed or has code. Cite them as background only, and do not rely on their numbers. |
| **Guidance Watermarking, arXiv:2509.22126** | **Exists. The venue is CORRECT: ICLR 2026 poster** (iclr.cc/virtual/2026/poster/10011427). Authors: Gesny, Giboulot, Furon, Chappelier. The method claim is correct: it guides diffusion sampling, without training, using gradients from any off-the-shelf post-hoc watermark decoder, with augmentations added for robustness. **It covers generation only, not editing an image that already carries a watermark.** I found no public code (UNVERIFIED). |
| RIW, 2311.13713 | Exists: "A Somewhat Robust Image Watermark against Diffusion-based Editing Models" (Tan et al., 2023). I could not confirm a venue, so treat it as an arXiv preprint. It optimizes each image with PGD and was tested against IP2P and Imagic, keeping about 96% after IP2P. Embedding takes about 95 s per image. The paper points to an anonymous code repository (UNVERIFIED). |
| Tree-Ring, 2305.20030 | Correct. NeurIPS 2023 (from knowledge). Code: YuxinWenRick/tree-ring-watermark (MIT). |
| Gaussian Shading, 2404.04956 | Correct. CVPR 2024. Code: bsmhmmlf/Gaussian-Shading (MIT). |
| ZoDiac, 2401.04247 | Correct: "Attack-Resilient Image Watermarking Using Stable Diffusion", NeurIPS 2024. Code: zhanglijun95/ZoDiac (no LICENSE file). |
| WAM, 2411.07231 | Correct. ICLR 2025. Code is MIT. The `wam_mit.pth` weights are MIT; the paper's `wam_coco.pth` weights are non-commercial. |
| Stable Signature, 2303.15435 | Correct. ICCV 2023. The repository is CC BY-NC 4.0. |
| InstructPix2Pix, 2211.09800 | Correct. Code is MIT. |
| TurboEdit, 2408.00735 | Correct: Deutch et al., "TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models", SIGGRAPH Asia 2024. Code: GiilDe/turbo-edit (no LICENSE file). **Name collision:** arXiv:2408.08332 (Wu et al., ECCV 2024) is a *different* "TurboEdit". |
| PartEdit, 2502.04050 | Correct. SIGGRAPH 2025. Code: Gorluxor/part-edit (exists, no LICENSE file). |
| Falcon, github.com/SRDdev/Falcon | **No associated paper found.** The repo was created 2025-08-21 and has 0 stars. It wires WAM and InstructPix2Pix together as git submodules. The README says MIT, but the repo has **no LICENSE file**. The README also claims multi-iteration editing with detection after each step, but the committed `experiments/` folder holds only watermarking outputs (`watermarked.png`, `detection.json`, masks) and **no edit results**. |

## (b) What has already been measured

- **Single diffusion edits break most post-hoc watermarks.** W-Bench (VINE paper, ICLR 2025) tests 11 watermarks plus VINE on 10k images. It covers stochastic and deterministic regeneration, global edits (IP2P, MagicBrush, UltraEdit), local edits (ControlNet-Inpainting at several mask sizes) and image-to-video (SVD), scored by TPR@0.1%FPR. Zhao et al. (NeurIPS 2024), WAVES (ICML 2024) and CtrlRegen (ICLR 2025) measure removal by regeneration.
- **The strength of the edit matters.** SDEdit-style noise strength is in effect the regeneration-attack dial. The templated theory papers claim mutual information goes to zero as strength grows.
- **The editor matters, and newer editors damage watermarks less.** Evennou & Kijak (IH&MMSec 2026, 2604.25491) tested 8 editors from 2022 to 2026 with VideoSeal and TrustMark. Incidental removal peaked with Qwen-Image-Edit, around 2024, at 53-60%. FLUX 2 Klein and FireRed remove the watermark in under 10% of *local* edits. **This is a key confound.** Some of the "problem" may already be fading with modern editors.
- **Commercial editors launder watermarks with a single reconstruction prompt** (2609.01249, six OpenAI/Google editors).
- **Local watermarks under inpainting:** see 2609.16832 (AISec 2026). Generative inpainting hurts WAM and MaskWM even when it does not touch the watermarked region.
- **Sequential editing plus watermarks:** the only measurement I found is **Robust-Wide (ECCV 2024)**. It reports a "continual editing" figure for up to **3 rounds of InstructPix2Pix on 30 real images**, with BER rising each round but the watermark still extractable. That is qualitative and small. W-Bench, SafeMark, WAVES and 2604.25491 have **no sequential protocol**. 2609.16832 has only one classical chain (JPEG, then crop, then brightness). Multi-round editing benchmarks such as I2EBench2.0 (IJCV 2026, 2-5 rounds), MagicBrush multi-turn sessions and AnchorEdit measure edit quality, not watermarks.

## (c) What has already been solved, at least partly

- **Robust watermark embedding against editing (embedding side):** Robust-Wide (gradients through the last k IP2P steps), VINE-R, JigMark (black-box, contrastive), RIW (per-image adversarial), OmniGuard/EditGuard (AIGC-edit simulation) and ZoDiac (robust to regeneration). These change the *watermark*, not the editor.
- **Editor-side preservation that needs training:** SafeMark fine-tunes the editor with a watermark-decoding loss and reports bit accuracy near 1.0, but only on older domain editors.
- **Training-free embedding through decoder-gradient guidance:** Guidance Watermarking (ICLR 2026), for *generation* only.
- **Training-free recovery at detection time:** PGID (2605.09319) uses inversion-denoising cycles, but only for noise-based (Tree-Ring-type) watermarks.

## (d) What remains open

1. **Training-free, editor-side watermark preservation for modern text editors** (InstructPix2Pix, TurboEdit, ControlNet, PartEdit, SDEdit, FLUX-class). I found no method that takes an image *already* watermarked by an arbitrary post-hoc scheme (WAM, TrustMark, VINE) and changes the editor's sampling, without retraining, so that the decoded message survives. Guidance Watermarking shows the mechanism works for generation; SafeMark shows the goal is achievable with training. Combining them for editing is unpublished as far as I can find.
2. **A systematic sequential-edit protocol.** Robust-Wide's 3-round, 30-image figure is the only data point I found. Open: retention curves over 1 to 4+ rounds across editors, watermarks and edit types; whether loss compounds or levels off; whether re-guidance at each round stops the decay; and how edit quality drifts at the same time (CLIP-I/T, LPIPS to the original).
3. **Interaction with editor locality.** Localized editors (PartEdit, inpainting) against global ones (IP2P, SDEdit), combined with a localized watermark (WAM masks). Open questions: can preservation be limited to unedited regions, and does generative inpainting damage the mark outside the mask, as 2609.16832 found?
4. **Fair baselines that include modern editors.** Given 2604.25491, any claim that editing destroys watermarks has to be shown with the specific editors the study uses.
5. **Security trade-off.** A watermark-aware editor exposes decoder gradients. The same guidance can be flipped to remove or forge marks (2510.05978 describes a decoder-guided removal attack). No preservation paper has analyzed this.

## (e) Closest prior work and novelty

- **"Training-free watermark-preserving editing":** the closest papers are **SafeMark** (same goal, but it trains the editor, uses older editors and tests single edits) and **Guidance Watermarking** (same mechanism, but for generation, not editing). Next are Robust-Wide/VINE-R/JigMark, which make the watermark robust rather than the editor aware. **Novelty: moderate and real, but small in distance.** Decoder-gradient guidance inside an editor's sampling loop is an obvious next step from Guidance Watermarking, and someone could publish it at any time. Reviewers will cite both papers. The contribution must be the editing-specific parts: preserving an existing message without knowing the embedder, handling inversion-based and few-step editors, and limiting guidance to masked regions. It also needs comparison against SafeMark-style fine-tuning and an honest look at the removal/forgery dual-use.
- **"Watermark retention under sequential editing":** the closest is **Robust-Wide's continual-editing figure** (3 rounds, IP2P, 30 images). After that come W-Bench (single edit) and I2EBench2.0 (multi-round, no watermark). **Novelty: high for a systematic benchmark** (multiple watermarks, editors, rounds and a standard metric such as TPR@0.1%FPR plus bit accuracy). On its own it is a measurement contribution, though, and is stronger paired with the preservation method.
- **Overall:** the open contribution is (i) the first multi-round watermark-retention benchmark over modern text-based editors, plus (ii) a training-free, decoder-guided, watermark-aware editing wrapper that works with existing post-hoc watermarks, tested round by round.

## (f) Practical notes per watermark method

| Method | Works on arbitrary real images? | Detection metric | Pretrained weights | License | Repo | Without CUDA? |
|---|---|---|---|---|---|---|
| **WAM** | **Yes (post-hoc)**, with localized masks | Bit accuracy over 32 bits, detection p-value/TPR, mask IoU | `wam_mit.pth` (MIT), `wam_coco.pth` (non-commercial) | Code MIT | facebookresearch/watermark-anything | README was tested on CUDA 12.4. Plain PyTorch, so CPU/MPS inference should work (UNVERIFIED) |
| **TrustMark** | **Yes (post-hoc)**, any resolution | Bit accuracy (100 bits + ECC), detection | Downloaded automatically | MIT (code and models, per README) | adobe/trustmark | **Yes**, the README shows CPU mode |
| **VINE** (B/R) | **Yes (post-hoc)** | TPR@0.1%FPR, bit accuracy | On HuggingFace (Shilin-LU) | **NTU non-commercial** | Shilin-LU/VINE | Built on SDXL-Turbo, heavy. CPU impractical, MPS UNVERIFIED |
| **Stable Signature** | **No.** Only images decoded by the watermarked LDM VAE | Bit accuracy over 48 bits, TPR@FPR | Watermarked SD2 decoder + extractor | CC BY-NC 4.0 | facebookresearch/stable_signature | Extraction is a small CNN, so CPU is fine; generation needs SD |
| **Tree-Ring** | **No.** Watermarks *generations* through the initial noise; detection needs DDIM inversion with the same model | AUC, TPR@1%FPR (zero-bit, p-value) | Uses public SD | MIT | YuxinWenRick/tree-ring-watermark | Needs SD inversion; CPU/MPS slow (UNVERIFIED) |
| **Gaussian Shading** | **No.** Generation-only, inversion-based | TPR, bit accuracy | Uses public SD | MIT | bsmhmmlf/Gaussian-Shading | As for Tree-Ring |
| **RingID** | **No.** Generation-only | Identification accuracy | Public SD | No LICENSE file | showlab/RingID | As for Tree-Ring |
| **ZoDiac** | **Yes (post-hoc)**, by optimizing each image's latent to carry a Tree-Ring pattern | Watermark detection rate at FPR | Public SD | **No LICENSE file** | zhanglijun95/ZoDiac | Optimizing each image with SD makes CPU impractical |
| **RIW** | Yes (post-hoc, about 95 s/image PGD) | Custom extraction rates | SD VAE | UNVERIFIED | Anonymous repo (UNVERIFIED) | Impractical |
| **Robust-Wide** | Yes (post-hoc encoder) | BER (64 bits) | In repo (UNVERIFIED) | MIT | hurunyi/Robust-Wide | Encoder/decoder light; training needs IP2P |
| **InvisMark** | Yes (post-hoc) | Bit accuracy | UNVERIFIED | MIT | microsoft/InvisMark | UNVERIFIED |

**Confirmed:** Tree-Ring, Gaussian Shading and RingID (and Stable Signature) watermark *generated* images. They cannot be applied to real photos directly. ZoDiac is the post-hoc way to get a Tree-Ring-style watermark on real images. For a study of real-image editing, the practical post-hoc options are **WAM** (localized, MIT weights), **TrustMark** (MIT, runs on CPU) and **VINE** (the strongest against editing, but non-commercial and heavy).

Editors: InstructPix2Pix (MIT; diffusers pipeline), TurboEdit (GiilDe/turbo-edit, no license file, SDXL-Turbo), PartEdit (Gorluxor/part-edit, no license file), SDEdit (ermongroup/SDEdit, MIT), ControlNet (Apache-2.0).

## (g) Added after the pilot: the re-embedding baseline (2026-10-01)

- SafeMark (2605.19511) §3.1 assumes "an image editing service provider has access to both the editing model and
  the watermark encoder–decoder". Its text does **not** discuss or compare against decode → edit → re-embed
  (checked against the arXiv HTML; Appendix G only discusses white-/black-box editor access).
- Guidance Watermarking (2509.22126) is generation-only, so the question does not arise there.
- Our 10-image pilot (InstructPix2Pix, SDEdit; WAM, TrustMark, DWT-DCT-SVD): decode → edit → re-embed kept detection at
  its L0 level through 4 rounds with no measurable edit-fidelity cost (pilot is underpowered; stage 2 tests it at n=90).
- Consequence for framing: in any cooperative setting where the editor holds the codec, editor-side preservation must
  be compared against re-embedding. We do **not** reproduce SafeMark (no code; different editors/watermarks), so we make
  no claim about SafeMark's numbers — only that the baseline is missing from its evaluation.
