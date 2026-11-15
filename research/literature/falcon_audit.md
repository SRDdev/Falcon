# Falcon code audit

Repo: https://github.com/SRDdev/Falcon @ 44c2d26d937dd73b6b12b977d7853eb9dd6e8956 (2025-08-23). No LICENSE file in repo.

| Finding | Location | Impact |
|---|---|---|
| No post-edit watermark detection | pipeline/run_single.py | README claim unsupported by code |
| No sequential editing loop; only `samples[:2]`, prompt v1 | pipeline/run_single.py | Single edit only |
| Visible text (the edit prompt) drawn onto every edited image | models/editing/instruct_pix2pix.py `_add_watermark` | Would corrupt any post-edit metric — must be removed |
| WAM ground-truth message unseeded and not saved (only predicted msg stored) | models/watermarking/watermark_anything.py | Post-edit bit accuracy not computable from outputs |
| CUDA hardcoded (`.cuda()`, `autocast("cuda")`) | instruct_pix2pix.py | Not runnable on CPU/MPS |
| requirements.txt empty; hardcoded /home/shreyas paths; submodules (instructpix2pix, WAM) not vendored | repo root | Environment not reproducible as shipped |
| Uses original CompVis IP2P ckpt + k-diffusion Euler-ancestral, 100 steps, cfg_text 7.5, cfg_image 1.5, 512px | configs/config.yaml | Reusable settings; diffusers `timbrooks/instruct-pix2pix` is an equivalent, simpler loader |
| Dataset: SRDdev/AI-TextImageEdit (split V1), 3 edit prompts per image | data/dataset.py | Natural source for chained edit prompts |
| The "0.99 → 0.92 → 0.59" sequential numbers in RESEARCH_INVESTIGATION.md | — | Not produced by any code in repo; treat as unverified |

Reusable: WAM embed/detect call pattern, IP2P CFG denoiser. Everything else to be reimplemented cleanly.
