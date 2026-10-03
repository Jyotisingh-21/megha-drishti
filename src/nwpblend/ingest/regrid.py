import numpy as np
import xarray as xr


def regrid_dataset(ds: xr.Dataset, domain: dict) -> xr.Dataset:
    """
    Regrids a dataset to the target domain.
    Uses conservative-like area-averaging (via groupby_bins) for precipitation
    and bilinear interpolation for temperature/wind.

    If the raw resolution matches the domain resolution, it just interpolates.
    """
    target_lats = np.arange(domain["lat_max"], domain["lat_min"] - 0.01, -domain["resolution"])
    target_lons = np.arange(domain["lon_min"], domain["lon_max"] + 0.01, domain["resolution"])

    if "latitude" in ds.dims:
        lat_name, lon_name = "latitude", "longitude"
    else:
        lat_name, lon_name = "lat", "lon"

    # Check resolution
    raw_res = abs(float(ds[lat_name].values[0] - ds[lat_name].values[1]))
    target_res = domain["resolution"]

    # If resolutions are close, simple linear interpolation is sufficient for all
    if abs(raw_res - target_res) < 1e-4:
        return ds.interp({lat_name: target_lats, lon_name: target_lons}, method="linear")

    # Otherwise (e.g. 0.1 to 0.25), use area averaging for precip, linear for others
    # 1. Linear for t2m, wind
    linear_vars = [v for v in ds.data_vars if v not in ["tp", "prate", "precip"]]
    if linear_vars:
        ds_lin = ds[linear_vars].interp(
            {lat_name: target_lats, lon_name: target_lons}, method="linear"
        )
    else:
        ds_lin = xr.Dataset(coords={lat_name: target_lats, lon_name: target_lons})

    # 2. Area averaging for precip
    precip_vars = [v for v in ds.data_vars if v in ["tp", "prate", "precip"]]
    if precip_vars:
        half_res = target_res / 2.0
        lat_bins = np.append(target_lats + half_res, target_lats[-1] - half_res)
        lat_bins = np.sort(lat_bins)  # Must be increasing
        lon_bins = np.append(target_lons - half_res, target_lons[-1] + half_res)

        # We need to sort lats if they are descending for groupby_bins
        ds_p = ds[precip_vars]
        if ds_p[lat_name].values[0] > ds_p[lat_name].values[-1]:
            ds_p = ds_p.sortby(lat_name)

        g_lon = ds_p.groupby_bins(lon_name, lon_bins, labels=target_lons).mean(dim=lon_name)
        g_lat = g_lon.groupby_bins(lat_name, lat_bins, labels=target_lats[::-1]).mean(dim=lat_name)

        g_lat = g_lat.sortby(f"{lat_name}_bins", ascending=False)
        g_lat = g_lat.rename({f"{lat_name}_bins": lat_name, f"{lon_name}_bins": lon_name})

        # Merge
        ds_out = xr.merge([ds_lin, g_lat])
    else:
        ds_out = ds_lin

    return ds_out
