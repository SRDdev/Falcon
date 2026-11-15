# Falcon / PECCAVI-Edit — what exists, what changed, what is new

Branch `peccavi-edit-spotlight` (6 commits on top of `results-pages`, 31 files, +1,134 lines). Written 2026-10-27.
Target: CVPR 2027 — **enrol by Nov 10**, PDF Nov 16, supplement Nov 23. Results freeze Nov 9.

## 1. State before this branch (`results-pages`, Oct 2)

Done and verified; none of it is on `master` (which is still SRDdev's Aug 2025 prototype).

| Area | What exists | Where |
|---|---|---|
| Pipeline | Clean rewrite of Falcon: PIE-Bench++ loader, 3 editors (IP2P, SDEdit, FLUX.2-klein), 3 post-hoc watermarks (WAM, TrustMark, DWT-DCT-SVD) + 2 generation-time (Tree-Ring, Gaussian Shading), metrics, 4-round sequential editing, re-embed control, decoder-guided editing, probes; resumable runs, manifests | `src/wmedit/`, `run_*.py` |
| Rigour | Pre-registered metric definitions; paired design (same images/prompts/seeds); TPR@FPR 1e-3; McNemar/Wilcoxon + Holm; bootstrap CIs | `analysis/PREREGISTRATION.md`, `analysis/analyze.py` |
| Results | Stage 2 (90 images × 4 rounds × 3 editors): baseline detection collapses (WAM L4: IP2P 19%, klein 0%, SDEdit 0%); re-embed ≈ 100%; 0 false positives / 1,350. Guided editing on 18 held-out images: L4 72–89%. Forgery: guidance *creates* marks on 5–12/18 clean images. Generated-image track: Gaussian Shading survives IP2P (96%) not SDEdit (4%) | `tables/stage2_*.md`, `tables/guided_test_*.md`, `experiments/guided_test/forgery/rows.csv`, `tables/gentrack_*.md` |
| Literature | Claim-by-claim audit (caught a false venue claim and a templated-paper cluster); Falcon code audit | `literature/gap_analysis.md`, `literature/falcon_audit.md` |
| Site | GitHub Pages report | https://prajak002.github.io/Falcon/ |

Not done then: region analysis (script only), VINE/Robust-Wide/PECCAVI baselines, MagicBrush/UltraEdit/Kontext editors; `STATUS.md` stale.

## 2. New on this branch

### 2.1 Method code (CPU-tested, 12 tests pass: `pytest tests/test_erasure_ewo.py`)
- **`src/wmedit/erasure.py`** — PECCAVI-Edit Secs. 2.2–2.7: survival rates, co-failure **covariance**, box IoU, score fusion (Eq. 22), greedy failure-diverse carrier selection (Eq. 27), Poisson-binomial recovery, Cantelli bound (Eq. 45), **authenticated RS(K,R) erasure code** with keyed tag (Eqs. 28–40).
- **`src/wmedit/ewo.py`** — Edit–Watermark Overlap (Sec. 2.8): watermark energy, pixel edit map, **`regeneration_map`** (detects residual re-sampling where content is unchanged — new), energy/area EWO, binned recovery curves with bootstrap CIs.
- **`src/wmedit/editors.py`** — `QwenImageEdit` wrapper (20B, Apache-2.0; held-out VLM-conditioned editor; evaluation attack only).

### 2.2 Analyses and scripts
- **`analysis/universality.py`** (+ `tables/stage2_universality.md`): does edit magnitude explain survival across editors? Within-editor AUC 0.76–0.98; held-out calibration fails → visible change is the wrong variable. Source of the **invisible-edit paradox** (SDEdit: least pixel change, 100% removal).
- **`run_ewo_analysis.py`** — four EWO variants for every existing edit + R(ρ) curves (needs images on the GPU host).
- **`kaggle/run.py`, `kaggle/kernel-metadata.json`, `configs/kaggle.yaml`** — Kaggle T4 kernel: 27-image baseline (IP2P + SDEdit) → EWO + region + universality. Running: https://www.kaggle.com/code/popsensharma/peccavi-edit-kaggle27
- **`scripts/create_issues.py`** + **`issues/*.md`** — GitHub issues from front-mattered markdown; rewrites `#KEY` to real numbers.

### 2.3 Planning
- **`paper/spotlight_experiments.md`** — 14 experiments (X1–X14) mapped to paper tables/figures with status.
- **19 GitHub issues**, 3 milestones (M1 Oct 31 infra · M2 Nov 7 results · M3 Nov 10 enrol / Nov 16 PDF).

## 3. Corrections to the PECCAVI-Edit draft (now reflected in code; fix in the paper — #14)
1. **Payload**: RS(16,8) over GF(2⁸) = 8 info bytes = **48-bit message + 16-bit tag**, not "64-bit + tag".
2. **Eq. 23**: co-failure must be **Cov(Zᵢ,Zⱼ)** (what Var[N_surv], Eq. 44 needs), not the joint-failure rate, which penalises two weak *independent* carriers. Eq. 27 needs its argmax.
3. **Blind search inflates false positives**: the keyed tag must be counted and clean-image FPR measured (#3).
4. Conclusion is copied from the text paper; template is ACL (17 pp.) not CVPR (8 pp.); PECCAVI is CVPR 2026 not arXiv; unresolved `(?)` citations.

## 4. Ideas taken from reference papers (concepts only — no code or method components borrowed)
| Source | Concept adopted into PECCAVI-Edit | Issue |
|---|---|---|
| TurboEdit (arXiv:2408.00735) | **Effective regeneration depth τ(p)**: per-pixel diffusion-timestep equivalent of watermark destruction; **content vs regeneration split of EWO**. Editor-independent survival target for f_ψ | #18 |
| Qwen-Image-Edit model card | Its own *appearance* vs *semantic* edit split mirrors ρ_content/ρ_regen; strongest held-out editor (VLM-conditioned mechanism) | #6 |
| Awesome text-based-manipulation list (ends 2023) | Mechanisms turn over every ~18 months → define Q_edit by **mechanism**, not model | #6 |
| Adobe Firefly (+ GPT Image, Nano Banana, FLUX partners) | The real untrusted editor is a commercial UI → n=50 black-box supplement | #19 |

Reverted on request: a TurboEdit reimplementation (commit d735829 → 2e3afe6).

## 5. Issue map
| # | Title | Milestone |
|---|---|---|
| 1 | [X1] Universal survival law: EWO on stage 2 | M1 |
| 2 | WAM region analysis (inside/outside edit mask) | M1 |
| 3 | PECCAVI-Edit embedder/detector — **critical path** | M1 |
| 4 | [X3] Survival atlas: Z matrix, co-failure structure | M2 |
| 5 | [X5] Edit-Survival Predictor f_ψ, co-failure head g_ψ | M2 |
| 6 | Editor suite by mechanism (MagicBrush, InstructDiffusion, UltraEdit, Kontext, Qwen) | M1 |
| 7 | Baselines: VINE, Robust-Wide, PECCAVI, Stable Signature, ZoDiac | M1 |
| 8 | [X2] Stability paradox: NMP stability vs edit survival | M2 |
| 9 | [X6] Oracle placement upper bound | M2 |
| 10 | [X7] Multi-turn T=8, watermark half-life | M2 |
| 11 | [X8] ESR-hunting adaptive attacker | M2 |
| 12 | [X9] Regime map: untrusted vs cooperative editor | M3 |
| 13 | [X11] Standard distortions + W-Bench | M3 |
| 14 | Paper fixes — **critical path** | M3 |
| 15 | Compute: ≥4 GPUs Oct 28–Nov 9 — **critical path** | M1 |
| 16 | Refresh STATUS.md | M1 |
| 17 | **Enrol on OpenReview by Nov 10** — hard gate | M3 |
| 18 | [X13] Effective regeneration depth τ(p) | M2 |
| 19 | [X14] Commercial black-box editors, n=50 | M3 |

## 6. Status of #3 (embedder) — first real numbers
`src/wmedit/peccavi_edit.py` + `tables/peccavi_capacity_pie54.md`: on 54 PIE-Bench++ photos at 40.2 dB, **54/54 exact 48-bit recovery, 0/54 false positives (clean and wrong key)**, 52/54 after JPEG-50 and after destroying 4/16 carriers; weak spot: 3-px shift (37/54). Key design step: whitened matched-filter detection (first attempt without it: 9/54). VINE and Robust-Wide wrappers added (#7).

## 7. Still open / risks
- Nothing has run on a GPU yet except the Kaggle kernel (in progress). Stage 2 images live only on the L4 host.
- #3 survival under VAE round-trip and real edits is unmeasured; translation sync needs work.
- Compute: ~140 GPU-h for the main benchmark; one L4 or Kaggle's 30 h/week is not enough.
- Kaggle token was pasted in chat — rotate it after the project.
