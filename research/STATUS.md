# Status report (generated 2026-10-02 02:11)

Every number below is read from `experiments/*`. Partial runs are labelled PARTIAL.

## Progress

| experiment | edited images on disk | evaluated rows |
|---|---|---|
| guided_test | 533 | not yet |
| pilot | 560 | 700 |
| probes | 0 | not yet |
| stage2 | 2422 | not yet |

## Pilot (n=10 images, complete) — detection rate @ FPR 1e-3, L0 = watermarked original

| editor   | method   | watermark   |   L0 |   L1 |   L2 |   L3 |   L4 |
|:---------|:---------|:------------|-----:|-----:|-----:|-----:|-----:|
| ip2p     | baseline | dwtdct      | 0.70 | 0.20 | 0.00 | 0.00 | 0.00 |
| ip2p     | baseline | trustmark   | 1.00 | 0.60 | 0.50 | 0.20 | 0.10 |
| ip2p     | baseline | wam         | 1.00 | 0.50 | 0.20 | 0.20 | 0.20 |
| ip2p     | reembed  | dwtdct      | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| ip2p     | reembed  | trustmark   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ip2p     | reembed  | wam         | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sdedit   | baseline | dwtdct      | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| sdedit   | baseline | trustmark   | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sdedit   | baseline | wam         | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sdedit   | reembed  | dwtdct      | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| sdedit   | reembed  | trustmark   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sdedit   | reembed  | wam         | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Bit accuracy:

| editor   | method   | watermark   |    L0 |    L1 |    L2 |    L3 |    L4 |
|:---------|:---------|:------------|------:|------:|------:|------:|------:|
| ip2p     | baseline | dwtdct      | 0.850 | 0.588 | 0.481 | 0.500 | 0.503 |
| ip2p     | baseline | trustmark   | 0.998 | 0.704 | 0.664 | 0.597 | 0.547 |
| ip2p     | baseline | wam         | 1.000 | 0.725 | 0.609 | 0.609 | 0.591 |
| ip2p     | reembed  | dwtdct      | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 |
| ip2p     | reembed  | trustmark   | 0.998 | 0.998 | 0.997 | 0.997 | 0.996 |
| ip2p     | reembed  | wam         | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sdedit   | baseline | dwtdct      | 0.850 | 0.503 | 0.481 | 0.487 | 0.497 |
| sdedit   | baseline | trustmark   | 0.998 | 0.505 | 0.509 | 0.488 | 0.475 |
| sdedit   | baseline | wam         | 1.000 | 0.537 | 0.537 | 0.531 | 0.531 |
| sdedit   | reembed  | dwtdct      | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 |
| sdedit   | reembed  | trustmark   | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 |
| sdedit   | reembed  | wam         | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

False positives on the unwatermarked track (all editors, levels):

| detector   |   false_positives |   n |
|:-----------|------------------:|----:|
| dwtdct     |                 0 | 100 |
| trustmark  |                 0 | 100 |
| wam        |                 0 | 100 |

Edit fidelity, mean over rounds 1–4 (`none` = B0, editor without watermark):

| editor   | method   | watermark   |   edit_success |   clip_dir |   clip_t_gain |   dino |   lpips |
|:---------|:---------|:------------|---------------:|-----------:|--------------:|-------:|--------:|
| ip2p     | baseline | dwtdct      |          0.625 |      0.130 |         0.007 |  0.777 |   0.283 |
| ip2p     | baseline | none        |          0.675 |      0.113 |         0.008 |  0.814 |   0.274 |
| ip2p     | baseline | trustmark   |          0.600 |      0.138 |         0.006 |  0.776 |   0.298 |
| ip2p     | baseline | wam         |          0.575 |      0.125 |         0.005 |  0.749 |   0.296 |
| ip2p     | reembed  | dwtdct      |          0.550 |      0.122 |         0.005 |  0.776 |   0.296 |
| ip2p     | reembed  | trustmark   |          0.525 |      0.131 |         0.004 |  0.792 |   0.291 |
| ip2p     | reembed  | wam         |          0.500 |      0.103 |         0.003 |  0.745 |   0.306 |
| sdedit   | baseline | dwtdct      |          0.675 |      0.075 |         0.020 |  0.832 |   0.334 |
| sdedit   | baseline | none        |          0.550 |      0.059 |         0.020 |  0.830 |   0.331 |
| sdedit   | baseline | trustmark   |          0.650 |      0.072 |         0.020 |  0.820 |   0.333 |
| sdedit   | baseline | wam         |          0.625 |      0.072 |         0.020 |  0.830 |   0.328 |
| sdedit   | reembed  | dwtdct      |          0.600 |      0.054 |         0.020 |  0.829 |   0.345 |
| sdedit   | reembed  | trustmark   |          0.625 |      0.073 |         0.021 |  0.823 |   0.328 |
| sdedit   | reembed  | wam         |          0.650 |      0.059 |         0.020 |  0.826 |   0.335 |

Watermark imperceptibility at L0 (vs original):

| watermark   |   wm_psnr |   wm_ssim |
|:------------|----------:|----------:|
| dwtdct      |     39.96 |      0.99 |
| trustmark   |     42.55 |      0.99 |
| wam         |     39.57 |      0.98 |

Figures: `figures/pilot_fig2_retention_det_rate.png`, `figures/pilot_fig3_methods_ip2p_det_rate.png`, `figures/pilot_fig4_tradeoff_clip_dir.png`, `figures/pilot_fig7_qualitative_wam_ip2p.png`

## Failure-mode probes (complete; up to 27 images per probe)

`det` = detection rate; `res_corr` = correlation of the watermark residual before/after the transform (1 = residual intact, 0 = destroyed).

| probe           |   n |   det_dwtdct |   det_trustmark |   det_wam |   res_corr_dwtdct |   res_corr_trustmark |   res_corr_wam |
|:----------------|----:|-------------:|----------------:|----------:|------------------:|---------------------:|---------------:|
| ip2p_identity   |  27 |        0.074 |           0.815 |     0.556 |             0.116 |                0.222 |          0.067 |
| ip2p_imgcfg_1.0 |  27 |        0.000 |           0.185 |     0.074 |             0.031 |                0.068 |          0.014 |
| ip2p_imgcfg_1.5 |  27 |        0.111 |           0.630 |     0.481 |             0.089 |                0.153 |          0.061 |
| ip2p_imgcfg_2.5 |  27 |        0.111 |           0.889 |     0.815 |             0.170 |                0.284 |          0.117 |
| ip2p_imgcfg_4.0 |  27 |        0.000 |           0.963 |     0.889 |             0.187 |                0.277 |          0.116 |
| ip2p_steps_10   |  27 |        0.037 |           0.630 |     0.296 |             0.060 |                0.152 |          0.039 |
| ip2p_steps_25   |  27 |        0.074 |           0.630 |     0.481 |             0.095 |                0.153 |          0.057 |
| sdedit_0.1      |  27 |        0.111 |           0.593 |     0.741 |             0.172 |                0.238 |          0.112 |
| sdedit_0.2      |  27 |        0.000 |           0.037 |     0.333 |             0.130 |                0.163 |          0.073 |
| sdedit_0.3      |  27 |        0.037 |           0.037 |     0.074 |             0.093 |                0.121 |          0.047 |
| sdedit_0.5      |  27 |        0.000 |           0.000 |     0.000 |             0.061 |                0.064 |          0.020 |
| sdedit_0.7      |  27 |        0.000 |           0.000 |     0.000 |             0.029 |                0.039 |          0.010 |
| vae_roundtrip   |  27 |        0.296 |           1.000 |     0.963 |             0.237 |                0.380 |          0.178 |

Figure: `figures/probes_fig9_probes.png`
