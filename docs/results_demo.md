# Benchmark Results (DEMO Data)

Evaluation of single models vs baseline blenders.

| Model | Var | RMSE | MAE | Bias |
|-------|-----|------|-----|------|
| Single: ecmwf_ifs | t2m | 2.046 | 0.923 | 0.835 |
| Single: gfs | t2m | 2.046 | 0.923 | 0.833 |
| Single: aifs | t2m | 2.049 | 0.965 | 0.834 |
| Single: ncum_g | t2m | 2.048 | 0.924 | 0.835 |
| Single: graphcast | t2m | 2.051 | 0.966 | 0.833 |
| Single: pangu | t2m | 2.050 | 0.966 | 0.833 |
| Blend: Equal | t2m | 2.043 | 0.880 | 0.834 |
| Blend: EWA | t2m | 2.043 | 0.880 | 0.834 |
| Blend: BMA | t2m | 2.043 | 0.880 | 0.834 |
| Single: ecmwf_ifs | precip | 17.876 | 7.081 | -5.116 |
| Single: gfs | precip | 17.871 | 7.070 | -5.118 |
| Single: aifs | precip | 19.737 | 8.124 | -6.873 |
| Single: ncum_g | precip | 17.877 | 7.082 | -5.110 |
| Single: graphcast | precip | 19.741 | 8.126 | -6.878 |
| Single: pangu | precip | 19.727 | 8.120 | -6.874 |
| Blend: Equal | precip | 18.731 | 7.516 | -5.995 |
| Blend: EWA | precip | 18.652 | 7.429 | -5.890 |
| Blend: BMA | precip | 18.621 | 7.418 | -5.867 |

