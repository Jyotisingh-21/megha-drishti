import logging
import os
import shutil
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from nwpblend.ingest.common import get_cache_dir

logger = logging.getLogger(__name__)

def probe_and_fetch(target_date, domain, leads, variables):
    """
    Probes for the latest complete ECMWF run, archiving to data/archive.
    """
    try:
        from ecmwf.opendata import Client
    except ImportError:
        logger.error("ecmwf-opendata not installed.")
        return None, None

    # Load mirrors from config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    mirrors = config.get("ecmwf_mirrors", ["aws", "google", "ecmwf", "azure"])

    # Determine candidate datetimes
    if target_date == "latest":
        now = datetime.now(UTC)
        # latest possible candidate is 00Z or 12Z
        candidates = []
        for i in range(4): # look back up to 4 runs (2 days)
            t = now - timedelta(hours=i*12)
            candidates.append(t.replace(hour=(12 if t.hour >= 12 else 0), minute=0, second=0, microsecond=0))
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)] # Fixed to 00Z for specific dates

    # Map variables
    var_map = {"t2m": "2t", "precip": "tp", "wind10m": ["10u", "10v"], "gust10m": "10fg"}
    req_vars = []
    for v in variables:
        if isinstance(var_map.get(v), list):
            req_vars.extend(var_map[v])
        elif var_map.get(v):
            req_vars.append(var_map[v])

    if not req_vars:
        return None, None

    success_ds = None
    chosen_run = None

    for cand in candidates:
        date_str = cand.strftime("%Y%m%d")
        time_int = cand.hour
        
        archive_dir = f"data/archive/ecmwf/{cand.strftime('%Y%m%d_%H%M')}"
        os.makedirs(archive_dir, exist_ok=True)
        target_file = os.path.join(archive_dir, "raw.grib")

        if os.path.exists(target_file):
            logger.info(f"Found archived ECMWF data for run {cand}")
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.warning("Archived file invalid. Re-downloading.")
                os.remove(target_file)

        logger.info(f"Probing ECMWF run {cand}...")
        download_success = False
        
        for mirror in mirrors:
            try:
                # Omit resol to get default dynamic resolution handling (especially for Oct 2026 0p1 change)
                client = Client(source=mirror, model="ifs") 
                client.retrieve(
                    date=date_str,
                    time=time_int,
                    step=leads,
                    type="fc",
                    param=req_vars,
                    target=target_file,
                )
                download_success = True
                logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                break
            except Exception as e:
                logger.debug(f"Mirror {mirror} failed for {cand}: {e}")
                continue
                
        if download_success:
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.error(f"Failed to process downloaded file for {cand}")
        else:
            logger.info(f"Run {cand} not available on any mirror.")
            
    if chosen_run:
        logger.info(f"Selected ECMWF run: {chosen_run}")
        return success_ds, chosen_run
    else:
        logger.error("No valid ECMWF runs found in probing window.")
        return None, None

def _process_file(target_file, domain, init_dt):
    try:
        ds = xr.open_dataset(target_file, engine="cfgrib")
        
        # Determine actual resolution
        lat_res = abs(ds.latitude.values[1] - ds.latitude.values[0])
        _ = abs(ds.longitude.values[1] - ds.longitude.values[0])
        logger.info(f"Detected ECMWF raw resolution: {lat_res:.2f} deg")
        
        # Subsetting (with slight buffer for interpolation)
        lat_slice = slice(domain["lat_max"] + 1, domain["lat_min"] - 1)
        lon_slice = slice(domain["lon_min"] - 1, domain["lon_max"] + 1)
        
        # If latitude is reversed
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            lat_slice = slice(domain["lat_min"] - 1, domain["lat_max"] + 1)
            
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

        # Regrid to pipeline grid (0.25)
        new_lats = np.arange(domain["lat_max"], domain["lat_min"] - 0.01, -domain["resolution"])
        new_lons = np.arange(domain["lon_min"], domain["lon_max"] + 0.01, domain["resolution"])
        
        ds = ds.interp(latitude=new_lats, longitude=new_lons, method="linear")

        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        if "step" in ds.coords:
            ds = ds.rename({"step": "lead"})

        out_vars = {}
        if "2t" in ds:
            out_vars["t2m"] = ds["2t"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"
        elif "t2m" in ds:
            out_vars["t2m"] = ds["t2m"] - 273.15
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
            out_ds = out_ds.expand_dims({"time": [init_dt]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to process ECMWF data: {e}")
        return None
