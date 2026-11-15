# Setup

## Hardware used
- Main GPU host: AWS g6.xlarge, NVIDIA L4 23 GB, Ubuntu 24.04, isolated uv venv (Python 3.12, torch 2.11.0+cu128).
  Stage 2, probes, guided/forgery tests and the generated-image track all run here.
- The 10-image pilot ran on a vast.ai RTX 3060 12 GB (driver 595.84, same torch pin); that host has since been
  decommissioned. The pilot is used only to validate the pipeline, never in the main results.
  Peak GPU memory: InstructPix2Pix ≈ 3.7 GB, guided InstructPix2Pix ≈ 5.5 GB, metrics (CLIP ViT-L/14 + DINOv2-S + LPIPS) ≈ 2 GB.
- Analysis/figures also run on CPU (macOS M1 tested for unit tests).

## Environment
```bash
bash scripts/setup_remote.sh      # creates nothing global; installs into the active venv (/venv/main on vast.ai)
```
It installs `torch`/`torchvision` (cu128 wheels), `requirements.txt`, clones the external repositories and downloads
the WAM checkpoint. Versions used for every result are written automatically into each experiment's
`manifest_*.json` (Python, torch, CUDA, driver, GPU, package versions, code hash, external repo commits).

| Component | Source | Version / commit | License |
|---|---|---|---|
| Watermark Anything (WAM) | github.com/facebookresearch/watermark-anything | `2c08af04` (see external_repos/VERSIONS.txt), checkpoint `wam_mit.pth` | MIT (code + MIT checkpoint) |
| TrustMark | `pip install trustmark` (Adobe) | variant Q, ECC off, 100 bits | MIT |
| DWT-DCT-SVD | `pip install invisible-watermark` | mode `dwtDctSvd`, 32 bits | MIT |
| InstructPix2Pix | HF `timbrooks/instruct-pix2pix` | snapshot `31519b5c`, fp16 | MIT (weights: CreativeML OpenRAIL-M) |
| SDEdit backbone | HF `stable-diffusion-v1-5/stable-diffusion-v1-5` | snapshot `451f4fe1`, fp16 | CreativeML OpenRAIL-M |
| CLIP | open_clip `ViT-L-14` / `openai` | — | MIT |
| DINOv2 | HF `facebook/dinov2-small` | — | Apache-2.0 |
| PIE-Bench++ | HF `UB-CVML-Group/PIE_Bench_pp` | parquet V1 | CC-BY-SA-4.0 |
| Falcon (audited, not executed) | github.com/SRDdev/Falcon | `44c2d26d` | README says MIT, no LICENSE file |

No Hugging Face token is required (all models/datasets above are ungated).

## Not included and why
- **Tree-Ring / Gaussian Shading / RingID**: generation-time watermarks; they cannot be applied to PIE-Bench++
  photographs. A generated-image track was agreed but has not been run (needs SD-2.1 + DDIM-inversion detection).
- **ZoDiac**: per-image latent optimisation, minutes per image on an A100; not feasible on a 3060 at benchmark scale.
- **TurboEdit / PartEdit / FLUX-class editors**: SDXL/FLUX weights exceed the 12 GB budget or need gated access.
- **DWT-DCT (`dwtDct` mode)**: only 52% of its own clean embeds were detectable on 27 PIE-Bench++ images (no editing);
  replaced by `dwtDctSvd` (81.5%).
