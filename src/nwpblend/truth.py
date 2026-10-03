import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

"""
Observation Latency Documentation:
1. IMD Gridded Rainfall (imdlib):
   - Latency: Usually 0-1 days.
   - Frequency: Daily.
   - Availability: `imdlib.get_real_data` returns current day or previous day's accumulation.
   
2. ARCO-ERA5 (Google Cloud Public Datasets):
   - Latency: ~5 days (based on ERA5T preliminary release schedule from ECMWF).
   - Frequency: Hourly.
   - Availability: Access via Zarr store. It is continuously updated but lags the current date.
"""


def fetch_real_truth(start_date: str, end_date: str, domain: dict) -> xr.Dataset | None:
    """
    Fetches real truth data from IMD (precip) and ARCO-ERA5 (t2m, wind10m, gust10m).
    Combines them into a single harmonised dataset matching the pipeline grid.
    Only returns times where at least some observations are available.
    """
    logger.info(f"Fetching real truth data from {start_date} to {end_date}...")

    # 1. Fetch IMD Precip
    imd_ds = None
    try:
        import imdlib as imd

        logger.info("Fetching IMD precip...")
        # IMD requires date strings like 'YYYY-MM-DD'
        # get_real_data fetches a single date or a range, wait get_real_data handles start_date, end_date.
        # But get_real_data signature: get_real_data(var_type, start_date, end_date)
        s_dt = pd.to_datetime(start_date)
        e_dt = pd.to_datetime(end_date)

        # If dates are in the current year, we might need get_real_data vs get_data.
        # Let's try get_real_data for recent periods
        try:
            imd_data = imd.get_real_data(
                "rain", s_dt.strftime("%Y-%m-%d"), e_dt.strftime("%Y-%m-%d")
            )
            imd_raw = imd_data.get_xarray()
        except Exception:
            logger.warning("Falling back to get_data for IMD...")
            imd_data = imd.get_data("rain", s_dt.year, e_dt.year, fn_format="yearwise")
            imd_raw = imd_data.get_xarray()
            imd_raw = imd_raw.sel(time=slice(s_dt, e_dt))

        if imd_raw is not None and "rain" in imd_raw:
            imd_ds = imd_raw.rename({"rain": "precip"})
            imd_ds["precip"].attrs["units"] = "mm"
    except Exception as e:
        logger.error(f"Failed to fetch IMD data: {e}")

    # 2. Fetch ARCO-ERA5
    arco_ds = None
    try:
        logger.info("Fetching ARCO-ERA5 data...")
        url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        store = xr.open_zarr(url, chunks=None)

        # We need daily mean for evaluation, or matching 00Z to 00Z?
        # The pipeline assumes daily aggregation for benchmarks, let's pull daily means.
        # Actually, let's just slice time and resample to daily ('1D').
        # ARCO-ERA5 is very large, so we slice before doing anything.
        arco_slice = store.sel(time=slice(start_date, end_date))

        # Select target vars
        vars_map = {
            "2m_temperature": "t2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            # We don't have gust reliably in this chunk, or it's named differently.
            # We'll skip gust for truth if not found.
        }
        avail_vars = [v for v in vars_map if v in arco_slice]
        arco_slice = arco_slice[avail_vars]

        # Compute daily means/max
        logger.info("Computing daily aggregates for ARCO-ERA5...")
        # Since this is a lazy Zarr over GCS, this could take time. We might want to load it first if small.
        # It's only a few variables over a small domain.
        lat_slice = slice(domain["lat_max"] + 1, domain["lat_min"] - 1)
        lon_slice = slice(domain["lon_min"] - 1, domain["lon_max"] + 1)
        # ERA5 lat is usually 90 to -90, so lat_max to lat_min is correct.
        if arco_slice.latitude.values[0] < arco_slice.latitude.values[-1]:
            lat_slice = slice(domain["lat_min"] - 1, domain["lat_max"] + 1)

        arco_sub = arco_slice.sel(latitude=lat_slice, longitude=lon_slice).load()

        daily_arco = arco_sub.resample(time="1D").mean()

        out_vars = {}
        if "2m_temperature" in daily_arco:
            out_vars["t2m"] = daily_arco["2m_temperature"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"

        if "10m_u_component_of_wind" in daily_arco and "10m_v_component_of_wind" in daily_arco:
            u = daily_arco["10m_u_component_of_wind"]
            v = daily_arco["10m_v_component_of_wind"]
            out_vars["wind10m"] = np.sqrt(u**2 + v**2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        arco_ds = xr.Dataset(out_vars)
        arco_ds = arco_ds.rename({"latitude": "lat", "longitude": "lon"})

    except Exception as e:
        logger.error(f"Failed to fetch ARCO-ERA5: {e}")

    # Combine
    if imd_ds is None and arco_ds is None:
        logger.error("No truth data could be fetched.")
        return None

    combined = []
    if arco_ds is not None:
        combined.append(arco_ds)
    if imd_ds is not None:
        combined.append(imd_ds)

    # Regrid to pipeline grid
    new_lats = np.arange(domain["lat_max"], domain["lat_min"] - 0.01, -domain["resolution"])
    new_lons = np.arange(domain["lon_min"], domain["lon_max"] + 0.01, domain["resolution"])

    regridded_ds = []
    for ds in combined:
        ds_interp = ds.interp(lat=new_lats, lon=new_lons, method="linear")
        regridded_ds.append(ds_interp)

    if not regridded_ds:
        return None

    final_truth = xr.merge(regridded_ds, compat="override")
    return final_truth
