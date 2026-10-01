# Benchmark Results (DEMO Data)

Evaluation of single models vs baseline blenders.

| Model | Var | RMSE | MAE | Bias |
|-------|-----|------|-----|------|
| Single: ecmwf_ifs | t2m | 0.133 | 0.107 | 0.001 |
| Single: gfs | t2m | 0.134 | 0.108 | -0.000 |
| Single: aifs | t2m | 0.197 | 0.158 | 0.001 |
| Single: ncum_g | t2m | 0.134 | 0.108 | 0.002 |
| Single: graphcast | t2m | 0.197 | 0.159 | -0.001 |
| Single: pangu | t2m | 0.198 | 0.159 | 0.000 |
| Blend: Equal | t2m | 0.069 | 0.056 | 0.001 |
| Blend: EWA | t2m | 0.069 | 0.056 | 0.001 |
| Blend: BMA | t2m | 0.066 | 0.053 | 0.001 |
| Single: ecmwf_ifs | precip | 0.871 | 0.589 | 0.258 |
| Single: gfs | precip | 0.862 | 0.580 | 0.255 |
| Single: aifs | precip | 3.104 | 1.818 | -1.500 |
| Single: ncum_g | precip | 0.874 | 0.591 | 0.263 |
| Single: graphcast | precip | 3.104 | 1.821 | -1.505 |
| Single: pangu | precip | 3.098 | 1.815 | -1.501 |
| Blend: Equal | precip | 1.584 | 1.042 | -0.622 |
| Blend: EWA | precip | 1.430 | 0.929 | -0.502 |
| Blend: BMA | precip | 0.630 | 0.482 | 0.117 |
