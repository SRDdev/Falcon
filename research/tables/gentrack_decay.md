| method   | editor   | watermark       |   linear_b |   linear_sse |   exp_k |   exp_sse | better      |   first_round_drop |   mean_later_drop |
|:---------|:---------|:----------------|-----------:|-------------:|--------:|----------:|:------------|-------------------:|------------------:|
| baseline | ip2p     | gaussianshading |      0.055 |        0.000 |   0.060 |     0.000 | linear      |              0.044 |             0.056 |
| baseline | ip2p     | treering        |      1.910 |      nan     |   7.639 |   nan     | linear      |            nan     |           nan     |
| baseline | sdedit   | gaussianshading |      0.261 |        0.063 |   0.547 |     0.008 | exponential |              0.347 |             0.183 |
| baseline | sdedit   | treering        |      1.910 |      nan     |   7.639 |   nan     | linear      |            nan     |           nan     |