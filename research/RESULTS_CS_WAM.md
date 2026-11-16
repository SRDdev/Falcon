# CS-WAM Experimental Results: Multi-Turn Diffusion Edit Robustness

**Branch:** `feat/saliency-hybrid`  
**Config:** [`research/configs/saliency_hybrid.yaml`](configs/saliency_hybrid.yaml)  
**Evaluation Date:** 2026-11-16

---

## 1. Experimental Setup

- **Dataset:** PIE-Bench++ (`27` samples across 9 edit categories, `3` per subset, `seed = 42`).
- **Protocol:** Single-shot watermarking at $L_0$ (`method: baseline`, no re-embedding during edits), followed by **4 sequential untrusted AI edits** ($L_0 \rightarrow L_1 \rightarrow L_2 \rightarrow L_3 \rightarrow L_4$).
- **Editors:**
  - **InstructPix2Pix (`ip2p`):** `steps = 50`, `text_cfg = 7.5`, `image_cfg = 1.5`
  - **SDEdit (`sdedit`):** `steps = 50`, `strength = 0.5`, `cfg = 7.5`
- **Detection Criterion:** One-sided Binomial test at **$\text{FPR} = 10^{-3}$** (`26 / 32` matching bits for 32-bit methods; `66 / 100` for 100-bit TrustMark).
- **Total Evaluated Rows:** `1,620` (all `27` samples $\times$ `5` levels $\times$ `2` editors $\times$ `6` watermark tracks including `none`).

---

## 2. CS-WAM Method Summary

**Consensus-Saliency WAM (`cs_wam`)** ([`ConsensusSaliencyWAM`](src/wmedit/watermarks.py)) combines two components:

1. **Diffusion-Aware EOT Embedder:**
   - Warm-starts from WAM's neural embedder (`scaling_w = 2.0`), then refines the perturbation for `35` Adam steps (`eot_lr = 0.008`) bounded by $\ell_\infty \le 16/255$ (`eot_eps = 0.0627`, `eot_mse_weight = 40.0`).
   - Each step optimizes bit margin and localization mask survival through:
     - **SD-1.5 VAE round-trip** (`vae.encode` $\rightarrow$ `vae.decode`) with **latent Gaussian noise** ($\sigma \in [0.10, 0.45]$).
     - **Random 64×64 spatial tile dropout** (`25%` masked per step).
2. **Two-Stage Spatial Consensus Erasure Decoder:**
   - Computes bounded soft bit votes $v = \tanh(\ell / 2.0)$ and quadratic mask weights $w_p = m^2$ over an $8 \times 8$ grid (`64` tiles).
   - Uses **zero-diagonal pairwise cosine consensus** $A_{i,j} = \max(0, \cos(\mathbf{u}_i, \mathbf{u}_j))^2$ to locate the unedited cluster medoid and reject diffusion-corrupted outlier tiles.
   - Automatically falls back to global pooling when mutual cluster centrality $< 2.5$.

---

## 3. Results

### 3.1 Detection Rate (`det_rate`, higher is better)

#### InstructPix2Pix (`ip2p`)

| Watermark | $L_0$ | $L_1$ | $L_2$ | $L_3$ | $L_4$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`cs_wam` (Ours)** | **1.000** | **0.815** | **0.593** | **0.370** | **0.333** |
| `trustmark` | 1.000 | 0.593 | 0.444 | 0.259 | 0.185 |
| `wam` | 1.000 | 0.481 | 0.370 | 0.333 | 0.259 |
| `wam_saliency` | 1.000 | 0.333 | 0.185 | 0.074 | 0.111 |
| `peccavi_hybrid` | 0.889 | 0.037 | 0.000 | 0.000 | 0.000 |

#### SDEdit (`sdedit`, `strength = 0.5`)

| Watermark | $L_0$ | $L_1$ | $L_2$ | $L_3$ | $L_4$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cs_wam` (Ours) | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `trustmark` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `wam` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `wam_saliency` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `peccavi_hybrid` | 0.889 | 0.000 | 0.000 | 0.000 | 0.000 |

---

### 3.2 Bit Accuracy (`bit_acc`, higher is better)

#### InstructPix2Pix (`ip2p`)

| Watermark | $L_0$ | $L_1$ | $L_2$ | $L_3$ | $L_4$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`cs_wam` (Ours)** | 0.994 | **0.904** | **0.803** | **0.685** | 0.645 |
| `wam` | **1.000** | 0.736 | 0.669 | 0.663 | **0.652** |
| `trustmark` | 0.999 | 0.718 | 0.661 | 0.613 | 0.574 |
| `wam_saliency` | 0.997 | 0.683 | 0.628 | 0.558 | 0.552 |
| `peccavi_hybrid` | 0.967 | 0.590 | 0.588 | 0.553 | 0.529 |

#### SDEdit (`sdedit`, `strength = 0.5`)

| Watermark | $L_0$ | $L_1$ | $L_2$ | $L_3$ | $L_4$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`cs_wam` (Ours)** | 0.994 | **0.538** | **0.536** | **0.524** | **0.529** |
| `trustmark` | 0.999 | 0.519 | 0.484 | 0.497 | 0.501 |
| `wam` | **1.000** | 0.512 | 0.507 | 0.501 | 0.510 |
| `wam_saliency` | 0.997 | 0.491 | 0.502 | 0.520 | 0.503 |
| `peccavi_hybrid` | 0.967 | 0.508 | 0.502 | 0.495 | 0.510 |

---

### 3.3 False Positive Rate Control (`watermark = none`)

- All evaluated detectors recorded **`0` false positives across `270` unwatermarked images** ($\text{FPR} = 0.000$).

---

## 4. Key Findings

1. **Substantial Multi-Turn `ip2p` Survival Gain:** `cs_wam` achieves **81.5%** detection at $L_1$ (**+33.4 pp** over `wam`, **+22.2 pp** over `trustmark`) and **59.3%** at $L_2$ (**+22.3 pp** over `wam`, **+14.9 pp** over `trustmark`), while maintaining the highest detection rate at $L_3$ (**37.0%**) and $L_4$ (**33.3%**).
2. **Bit Accuracy Boost:** `cs_wam` increases $L_1$ bit accuracy from `73.6%` (`wam`) to **`90.4%`** and $L_2$ bit accuracy from `66.9%` to **`80.3%`**.
3. **SDEdit (`strength = 0.5`) Challenge:** While `cs_wam` is the only method consistently above `52.4%–53.8%` bit accuracy across all four SDEdit rounds, full-image 50% Gaussian noising + re-denoising still pushes all single-shot pixel-space watermarks below the `81.25%` (`26/32`) detection threshold.

---

## 5. Reproduction

```bash
# Run CPU unit tests
pytest -m "not gpu" -q

# Run full experiment
python research/run_experiment.py --config research/configs/saliency_hybrid.yaml --no-image-metrics

# Generate summary tables
python research/analysis/analyze.py --exp research/experiments/saliency_hybrid
```
