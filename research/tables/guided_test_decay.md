| method      | editor   | watermark   |   linear_b |   linear_sse |   exp_k |   exp_sse | better      |   first_round_drop |   mean_later_drop |
|:------------|:---------|:------------|-----------:|-------------:|--------:|----------:|:------------|-------------------:|------------------:|
| baseline    | ip2p     | trustmark   |      0.258 |        0.177 |   0.633 |     0.025 | exponential |              0.590 |             0.086 |
| baseline    | ip2p     | wam         |      0.225 |        0.166 |   0.465 |     0.052 | exponential |              0.549 |             0.065 |
| guided_l005 | ip2p     | trustmark   |      0.123 |        0.046 |   0.163 |     0.033 | exponential |              0.318 |             0.036 |
| guided_l005 | ip2p     | wam         |      0.092 |        0.041 |   0.114 |     0.033 | exponential |              0.247 |             0.009 |
| guided_l02  | ip2p     | trustmark   |      0.085 |        0.088 |   0.106 |     0.078 | exponential |              0.321 |            -0.029 |
| guided_l02  | ip2p     | wam         |      0.081 |        0.036 |   0.098 |     0.030 | exponential |              0.191 |             0.015 |
| reembed     | ip2p     | trustmark   |      0.001 |        0.000 |   0.001 |     0.000 | exponential |              0.001 |             0.000 |
| reembed     | ip2p     | wam         |      0.001 |        0.000 |   0.001 |     0.000 | linear      |              0.000 |             0.002 |