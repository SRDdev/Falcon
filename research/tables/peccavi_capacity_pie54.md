# PECCAVI-Edit capacity check pie54 (n=54, target PSNR 40.0 dB, K=16, RS(16,8), 48-bit msg)

Fidelity: PSNR 40.23 ± 0.43 dB

Reliability eta: watermarked regions 4.637, clean regions 0.587, erasure threshold 1.05

False positives: clean/no-watermark 0/54, wrong key 0/54

| attack        |      n |   bit_acc |   detected |   erased |   eta |
|:--------------|-------:|----------:|-----------:|---------:|------:|
| none          | 54.000 |     0.992 |      1.000 |    0.630 | 4.637 |
| jpeg90        | 54.000 |     0.991 |      1.000 |    0.704 | 4.391 |
| jpeg70        | 54.000 |     0.990 |      1.000 |    0.796 | 3.837 |
| jpeg50        | 54.000 |     0.988 |      0.963 |    1.204 | 3.139 |
| resize0.75    | 54.000 |     0.990 |      1.000 |    0.370 | 4.831 |
| blur1         | 54.000 |     0.990 |      0.963 |    0.222 | 5.069 |
| noise8        | 54.000 |     0.984 |      0.944 |    2.167 | 1.834 |
| shift3px      | 54.000 |     0.730 |      0.685 |    3.000 | 3.006 |
| erase4regions | 54.000 |     0.980 |      0.963 |    3.889 | 3.668 |
