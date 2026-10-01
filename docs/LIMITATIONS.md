# System Limitations & Hackathon Boundaries

In the spirit of scientific transparency (as mandated by rule #3 of our operational directives), the following limitations explicitly delineate the boundary between this functional prototype and a deployment-ready operational system.

## 1. Mocked Data & Demo Mode
- **Truth Proxy**: Due to lack of real-time authenticated access to IMD/GPM data during this sprint, the `--demo` mode heavily utilizes synthetic distributions modeled on Gamma (precipitation) and Gaussian (temperature) curves. 
- **Skill Metrics**: Skill percentages and ablation improvements demonstrated in `docs/results_demo.md` and the dashboard are reflective of the *math operating correctly on synthetic noise*, not real meteorological improvements. Targets like "10-15% lower RMSE" remain unverified on live atmospheric states.

## 2. Infrastructure Access
- **NCMRWF Integration**: The `NCUMAdapter` was built to strictly adhere to standard GRIB2 dimensions, but due to internal MoES firewall restrictions, it has not been tested against a live NCMRWF endpoint. It currently intercepts and safely skips execution via the `demo` mode logic.

## 3. Storage and Scaling
- **GRIB2 Export**: The native output uses highly compressed CF-1.8 NetCDF4. Writing multi-field GRIB2 via Python `eccodes` requires heavy C-compilation environments and strict template matching. The code currently stubs GRIB2 generation to avoid breaking Docker builds on non-Ubuntu environments.
- **In-Memory Operations**: While `xarray` and `zarr` support `dask`, the FastAPI endpoints currently read spatial grids synchronously. For a high-load national scale dashboard, a dedicated tiling server (e.g., TiTiler/PostGIS) would be required.

## 4. Probabilistic Rainfall
- **Two-Stage Hurdle Model**: The second stage of the extreme rain model (predicting amount conditional on rain) is currently using a deterministic GBM regressor. For true full PDF forecasting, it should be upgraded to predict the shape and scale parameters of a conditional Gamma distribution natively.
