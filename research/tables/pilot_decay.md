| method   | editor   | watermark   |   linear_b |   linear_sse |   exp_k |   exp_sse | better      |   first_round_drop |   mean_later_drop |
|:---------|:---------|:------------|-----------:|-------------:|--------:|----------:|:------------|-------------------:|------------------:|
| baseline | ip2p     | dwtdct      |      0.327 |        0.439 |   1.567 |     0.011 | exponential |              0.750 |             0.080 |
| baseline | ip2p     | trustmark   |      0.266 |        0.150 |   0.649 |     0.019 | exponential |              0.590 |             0.105 |
| baseline | ip2p     | wam         |      0.258 |        0.201 |   0.654 |     0.025 | exponential |              0.550 |             0.090 |
| baseline | sdedit   | dwtdct      |      0.341 |        0.687 |   4.821 |     0.004 | exponential |              0.991 |             0.006 |
| baseline | sdedit   | trustmark   |      0.341 |        0.610 |   4.565 |     0.003 | exponential |              0.990 |             0.020 |
| baseline | sdedit   | wam         |      0.311 |        0.563 |   2.425 |     0.012 | exponential |              0.925 |             0.004 |
| reembed  | ip2p     | dwtdct      |      0.000 |        0.000 |   0.000 |     0.000 | linear      |              0.000 |             0.000 |
| reembed  | ip2p     | trustmark   |      0.001 |        0.000 |   0.001 |     0.000 | linear      |              0.000 |             0.001 |
| reembed  | ip2p     | wam         |      0.000 |        0.000 |   0.000 |     0.000 | linear      |              0.000 |             0.000 |
| reembed  | sdedit   | dwtdct      |      0.000 |        0.000 |   0.000 |     0.000 | linear      |              0.000 |             0.000 |
| reembed  | sdedit   | trustmark   |      0.001 |        0.000 |   0.001 |     0.000 | exponential |              0.002 |             0.000 |
| reembed  | sdedit   | wam         |      0.000 |        0.000 |   0.000 |     0.000 | linear      |              0.000 |             0.000 |