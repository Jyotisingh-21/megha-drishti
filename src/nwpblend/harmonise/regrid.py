import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

try:
    import xesmf as xe

    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False
    logger.warning("xESMF is not available. Falling back to xarray interp (bilinear).")


def wrap_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """Wrap longitudes from 0..360 to -180..180 if needed."""
    if "lon" in ds.coords:
        # If any longitude is > 180, we assume it's 0-360 format and shift it.
        if ds.lon.max() > 180:
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
            ds = ds.sortby("lon")
    elif "longitude" in ds.coords and ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
    return ds


def _build_bounds(ds: xr.Dataset, coord: str) -> np.ndarray:
    """Helper to build 1D bounds for xESMF conservative regridding."""
    centers = ds[coord].values
    diffs = np.diff(centers)
    bounds = np.zeros(len(centers) + 1)
    # Assume regular grid
    d = diffs[0] if len(diffs) > 0 else 0.25
    bounds[:-1] = centers - d / 2.0
    bounds[-1] = centers[-1] + d / 2.0
    return bounds


def regrid(ds: xr.Dataset, target_grid: xr.Dataset) -> xr.Dataset:
    """
    Regrid dataset to target_grid.
    Uses conservative for precip, bilinear for others via xESMF.
    Falls back to xarray interp if unavailable.
    """
    ds = wrap_longitudes(ds)

    # Ensure target_grid has proper coordinate names
    t_lat = target_grid["lat"] if "lat" in target_grid.coords else target_grid["latitude"]
    t_lon = target_grid["lon"] if "lon" in target_grid.coords else target_grid["longitude"]

    if HAS_XESMF:
        # Separate precip (conservative) and other variables (bilinear)
        precip_vars = [
            v
            for v in ds.data_vars
            if "precip" in v.lower() or "tp" in v.lower() or "prate" in v.lower()
        ]
        other_vars = [v for v in ds.data_vars if v not in precip_vars]

        out_ds = xr.Dataset(coords={"lat": t_lat, "lon": t_lon})
        if "time" in ds.coords:
            out_ds = out_ds.assign_coords(time=ds.time)
        if "lead" in ds.coords:
            out_ds = out_ds.assign_coords(lead=ds.lead)

        # Prepare bounds for conservative regridding
        ds_bnds = ds.copy()
        target_bnds = xr.Dataset({"lat": t_lat, "lon": t_lon})

        s_lat = ds["lat"] if "lat" in ds.coords else ds["latitude"]
        s_lon = ds["lon"] if "lon" in ds.coords else ds["longitude"]

        ds_bnds["lat_b"] = (["lat_b"], _build_bounds(ds_bnds, s_lat.name))
        ds_bnds["lon_b"] = (["lon_b"], _build_bounds(ds_bnds, s_lon.name))
        target_bnds["lat_b"] = (["lat_b"], _build_bounds(target_bnds, "lat"))
        target_bnds["lon_b"] = (["lon_b"], _build_bounds(target_bnds, "lon"))

        if precip_vars:
            regridder_cons = xe.Regridder(
                ds_bnds, target_bnds, "conservative", unmapped_to_nan=True
            )
            for v in precip_vars:
                out_ds[v] = regridder_cons(ds[v])

        if other_vars:
            regridder_bil = xe.Regridder(ds, target_bnds, "bilinear", unmapped_to_nan=True)
            for v in other_vars:
                out_ds[v] = regridder_bil(ds[v])

        return out_ds
    else:
        # Fallback to linear
        return ds.interp(
            coords={s_lat.name: t_lat, s_lon.name: t_lon},
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
