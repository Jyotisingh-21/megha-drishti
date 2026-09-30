import logging

import xarray as xr

try:
    import xesmf as xe

    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False
    logger = logging.getLogger(__name__)
    logger.warning("xESMF is not available. Falling back to xarray interp (bilinear).")


def regrid(ds: xr.Dataset, target_grid: xr.Dataset) -> xr.Dataset:
    if HAS_XESMF:
        regridder = xe.Regridder(ds, target_grid, "conservative")
        return regridder(ds)
    else:
        return ds.interp(coords={"lat": target_grid.lat, "lon": target_grid.lon}, method="linear")
