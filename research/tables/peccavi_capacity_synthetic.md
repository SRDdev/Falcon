# PECCAVI-Edit capacity check synthetic (n=20, target PSNR 40.0 dB, K=16, RS(16,8), 48-bit msg)

Fidelity: PSNR 39.94 ± 0.00 dB

Reliability eta: watermarked regions 2.888, clean regions 1.066, erasure threshold 1.6

False positives: clean/no-watermark 0/20, wrong key 0/20

| attack        |      n |   bit_acc |   detected |   erased |   eta |
|:--------------|-------:|----------:|-----------:|---------:|------:|
| none          | 20.000 |     0.998 |      1.000 |    0.000 | 2.888 |
| jpeg90        | 20.000 |     0.998 |      1.000 |    0.000 | 2.852 |
| jpeg70        | 20.000 |     0.998 |      1.000 |    0.000 | 2.786 |
| jpeg50        | 20.000 |     0.999 |      1.000 |    0.000 | 2.715 |
| resize0.75    | 20.000 |     0.998 |      1.000 |    0.000 | 2.955 |
| blur1         | 20.000 |     0.999 |      1.000 |    0.000 | 3.383 |
| noise8        | 20.000 |     0.998 |      0.800 |    6.100 | 1.662 |
| shift3px      | 20.000 |     0.890 |      0.900 |    3.300 | 2.353 |
| erase4regions | 20.000 |     0.996 |      1.000 |    4.000 | 2.325 |
