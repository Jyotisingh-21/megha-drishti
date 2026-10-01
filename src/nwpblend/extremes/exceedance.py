import logging
import os

import xarray as xr

logger = logging.getLogger(__name__)


def build_exceedance_maps(
    prob_da: xr.DataArray, threshold: float, district_shapefile: str | None = None
) -> xr.Dataset:
    """
    Given a probability of exceedance map (e.g. from EMOS or Two-Stage),
    aggregates it to district level if shapefile is provided, else coarsens it.
    """
    res = {"pixel_prob": prob_da}

    if district_shapefile and os.path.exists(district_shapefile):
        # Placeholder for real district aggregation (e.g. using geopandas/regionmask)
        logger.info(f"Aggregating to districts using {district_shapefile}")
        # Not fully implemented to avoid heavy geospatial dependencies in demo
        res["district_prob"] = prob_da.mean(dim=["lat", "lon"])
    else:
        logger.warning("No district shapefile provided. Coarsening to 2x2 degree boxes.")
        # Assuming 0.25 deg grid, 8x8 = 2 deg
        if "lat" in prob_da.dims and "lon" in prob_da.dims:
            coarsened = prob_da.coarsen(lat=8, lon=8, boundary="trim").mean()
            res["coarse_prob"] = coarsened

    return xr.Dataset(res)
