import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def fetch(
    date: str,
    domain: dict,
    leads: list[int],
    variables: list[str],
    data_dir: str = "data/raw/ncmrwf",
) -> xr.Dataset | None:
    """
    Adapter for NCMRWF (NCUM-G / NEPS) GRIB2 files.
    Reads from a local directory.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.error(f"NCMRWF data directory {data_dir} is missing or empty.")
        return None

    dt = pd.to_datetime(date)
    date_str = dt.strftime("%Y%m%d")

    # Simple logic to find the file
    target_file = os.path.join(data_dir, f"ncmrwf_{date_str}.grib2")
    if not os.path.exists(target_file):
        logger.error(f"NCMRWF file {target_file} not found.")
        return None

    try:
        ds = xr.open_dataset(target_file, engine="cfgrib")
        ds = ds.sel(
            latitude=slice(domain["lat_max"], domain["lat_min"]),
            longitude=slice(domain["lon_min"], domain["lon_max"]),
        )

        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        if "step" in ds.coords:
            ds = ds.rename({"step": "lead"})

        out_vars = {}
        # Assuming NCUM uses names similar to others for this adapter:
        if "t2m" in ds:
            out_vars["t2m"] = ds["t2m"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"

        if "tp" in ds:
            out_vars["precip"] = ds["tp"]  # Assuming already in mm
            out_vars["precip"].attrs["units"] = "mm"

        if "u10" in ds and "v10" in ds:
            out_vars["wind10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        if "gust" in ds:
            out_vars["gust10m"] = ds["gust"]
            out_vars["gust10m"].attrs["units"] = "m s-1"

        out_ds = xr.Dataset(out_vars)
        if "time" not in out_ds.coords:
            out_ds = out_ds.expand_dims({"time": [dt]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to process NCMRWF data: {e}")
        return None
