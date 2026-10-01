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
    Fetch NOAA GFS data from anonymous AWS S3.
    """
    try:
        import s3fs
    except ImportError:
        logger.error("s3fs not installed.")
        return None

    cache_dir = get_cache_dir("gfs")
    dt = pd.to_datetime(date)
    date_str = dt.strftime("%Y%m%d")

    fs = s3fs.S3FileSystem(anon=True)

    datasets = []

    for lead in leads:
        lead_str = f"{lead:03d}"
        s3_path = f"noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead_str}"
        target_file = os.path.join(cache_dir, f"gfs_{date_str}_f{lead_str}.grib")

        if not os.path.exists(target_file):
            logger.info(f"Downloading GFS data for {date} lead {lead} to {target_file}")
            try:
                fs.get(s3_path, target_file)
            except Exception as e:
                logger.error(f"GFS download failed for lead {lead}: {e}")
                continue

        try:
            ds = xr.open_dataset(
                target_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface"}
            )
            ds_subset = ds.sel(
                latitude=slice(domain["lat_max"], domain["lat_min"]),
                longitude=slice(domain["lon_min"], domain["lon_max"]),
            )
            datasets.append(ds_subset)
        except Exception as e:
            logger.error(f"Failed to process GFS data for lead {lead}: {e}")
            continue

    if not datasets:
        return None

    try:
        combined = xr.concat(datasets, dim="step")
        combined = combined.rename({"latitude": "lat", "longitude": "lon", "step": "lead"})

        out_vars = {}
        # GFS uses different variable names, e.g., t2m, prate
        if "t2m" in combined:
            out_vars["t2m"] = combined["t2m"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"

        # prate is in kg m-2 s-1, convert to mm over 24h: prate * 86400
        # Actually GFS might provide PRATE or APCP. If it's prate:
        if "prate" in combined:
            out_vars["precip"] = combined["prate"] * 86400.0
            out_vars["precip"].attrs["units"] = "mm"

        if "u10" in combined and "v10" in combined:
            out_vars["wind10m"] = np.sqrt(combined["u10"] ** 2 + combined["v10"] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        if "gust" in combined:
            out_vars["gust10m"] = combined["gust"]
            out_vars["gust10m"].attrs["units"] = "m s-1"

        out_ds = xr.Dataset(out_vars)
        if "time" not in out_ds.coords:
            out_ds = out_ds.expand_dims({"time": [dt]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to combine GFS data: {e}")
        return None
