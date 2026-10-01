import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.ingest.common import get_cache_dir, retry_with_backoff

logger = logging.getLogger(__name__)


@retry_with_backoff(retries=3)
def fetch(date: str, domain: dict, leads: list[int], variables: list[str]) -> xr.Dataset | None:
    """
    Fetch ECMWF Open Data (HRES).
    """
    try:
        from ecmwf.opendata import Client
    except ImportError:
        logger.error("ecmwf-opendata not installed.")
        return None

    cache_dir = get_cache_dir("ecmwf")
    client = Client(source="ecmwf", model="ifs", resol="0p25")

    dt = pd.to_datetime(date)

    # Map variables
    var_map = {"t2m": "2t", "precip": "tp", "wind10m": ["10u", "10v"], "gust10m": "10fg"}

    req_vars = []
    for v in variables:
        if isinstance(var_map.get(v), list):
            req_vars.extend(var_map[v])
        elif var_map.get(v):
            req_vars.append(var_map[v])

    if not req_vars:
        return None

    target_file = os.path.join(cache_dir, f"ecmwf_{date}.grib")

    if not os.path.exists(target_file):
        logger.info(f"Downloading ECMWF data for {date} to {target_file}")
        try:
            client.retrieve(
                date=dt.strftime("%Y%m%d"),
                time=0,
                step=leads,
                type="fc",
                param=req_vars,
                target=target_file,
            )
        except Exception as e:
            logger.error(f"ECMWF download failed: {e}")
            return None

    if not os.path.exists(target_file):
        return None

    try:
        # Load and convert
        ds = xr.open_dataset(target_file, engine="cfgrib")
        ds = ds.sel(
            latitude=slice(domain["lat_max"], domain["lat_min"]),
            longitude=slice(domain["lon_min"], domain["lon_max"]),
        )

        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        if "step" in ds.coords:
            # step is timedelta64[ns], keep it
            ds = ds.rename({"step": "lead"})

        out_vars = {}
        if "2t" in ds:
            out_vars["t2m"] = ds["2t"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"

        if "tp" in ds:
            out_vars["precip"] = ds["tp"] * 1000.0  # m to mm
            out_vars["precip"].attrs["units"] = "mm"

        if "10u" in ds and "10v" in ds:
            out_vars["wind10m"] = np.sqrt(ds["10u"] ** 2 + ds["10v"] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        if "10fg" in ds:
            out_vars["gust10m"] = ds["10fg"]
            out_vars["gust10m"].attrs["units"] = "m s-1"

        out_ds = xr.Dataset(out_vars)
        if "time" not in out_ds.coords:
            out_ds = out_ds.expand_dims({"time": [dt]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to process ECMWF data: {e}")
        return None
