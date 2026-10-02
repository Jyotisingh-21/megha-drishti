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
        for i in range(4):  # look back up to 4 runs (2 days)
            t = now - timedelta(hours=i * 12)
            candidates.append(
                t.replace(hour=(12 if t.hour >= 12 else 0), minute=0, second=0, microsecond=0)
            )
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)]  # Fixed to 00Z for specific dates

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
                # Apply strict caps on retries to fail fast and move to next mirror
                client = Client(
                    source=mirror,
                    model="ifs",
                    maximum_retries=3,
                    retry_after=5,
                    use_server_retry_after=False,
                )

                # Wrap in timeout for hanging connections
                def do_retrieve(c, d_str, t_int, t_file):
                    c.retrieve(
                        date=d_str,
                        time=t_int,
                        step=leads,
                        type="fc",
                        param=req_vars,
                        target=t_file,
                    )

                import concurrent.futures

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(do_retrieve, client, date_str, time_int, target_file)
                # Configurable timeout
                download_timeout = config.get("download_timeout", 600)
                future.result(timeout=download_timeout)

                download_success = True
                logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                break
            except Exception as e:
                # Handle 503 or timeout silently as fail-fast trigger
                if (
                    "503" in str(e)
                    or "Timeout" in str(e)
                    or isinstance(e, concurrent.futures.TimeoutError)
                ):
                    logger.warning(f"Mirror {mirror} throttled/timed out. Switching immediately.")
                else:
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
    """Process a downloaded ECMWF GRIB file into the pipeline schema.

    Opens the GRIB with separate cfgrib filter_by_keys calls per variable
    group to avoid DatasetBuildError when variables live at different
    heightAboveGround levels (2 m vs 10 m) or have different step shapes.
    """
    try:
        # --- Open each variable group separately ---------------------------
        groups = [
            {"shortName": "2t"},
            {"shortName": "tp"},
            {"shortName": "10u"},
            {"shortName": "10v"},
            {"shortName": "10fg"},
        ]

        parts = {}
        primary_step = None  # The step coordinate from the first successfully opened variable
        for filt in groups:
            try:
                part = xr.open_dataset(
                    target_file,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": filt},
                )
                # Drop scalar heightAboveGround so merge doesn't conflict
                if "heightAboveGround" in part.coords:
                    part = part.drop_vars("heightAboveGround")

                # Track the primary step shape (from the first variable that has it)
                if primary_step is None and "step" in part.dims:
                    primary_step = part.step

                # Store all data variables from this part
                for vname in part.data_vars:
                    parts[vname] = part[vname]
            except Exception:
                pass  # Variable not present in this file

        if not parts:
            logger.error("No recognised variables found in ECMWF GRIB file")
            return None

        # Build the merged dataset, aligning step dimensions.
        # Variables with a scalar or mismatched step are broadcast/selected
        # to match the primary step shape.
        aligned = {}
        for vname, da in parts.items():
            if "valid_time" in da.coords:
                da = da.drop_vars("valid_time")
                
            if "step" in da.dims and primary_step is not None:
                # Re-index to primary step, filling with NaN where missing
                da = da.reindex(step=primary_step, method=None)
            elif primary_step is not None and "step" not in da.dims:
                # Scalar step – broadcast to all primary steps
                da = da.expand_dims(step=primary_step)
            aligned[vname] = da

        ds = xr.Dataset(aligned)

        # --- Determine actual resolution -----------------------------------
        lat_res = abs(float(ds.latitude.values[1]) - float(ds.latitude.values[0]))
        logger.info(f"Detected ECMWF raw resolution: {lat_res:.2f} deg")

        # --- Subset ---------------------------------------------------------
        lat_slice = slice(domain["lat_max"] + 1, domain["lat_min"] - 1)
        lon_slice = slice(domain["lon_min"] - 1, domain["lon_max"] + 1)
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            lat_slice = slice(domain["lat_min"] - 1, domain["lat_max"] + 1)
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

        # --- Regrid to pipeline grid ----------------------------------------
        new_lats = np.arange(domain["lat_max"], domain["lat_min"] - 0.01, -domain["resolution"])
        new_lons = np.arange(domain["lon_min"], domain["lon_max"] + 0.01, domain["resolution"])
        ds = ds.interp(latitude=new_lats, longitude=new_lons, method="linear")

        # --- Rename to canonical schema -------------------------------------
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        if "step" in ds.dims:
            ds = ds.rename({"step": "lead"})

        # --- Build output variables -----------------------------------------
        out_vars: dict = {}
        for name in ("2t", "t2m"):
            if name in ds:
                out_vars["t2m"] = ds[name] - 273.15
                out_vars["t2m"].attrs["units"] = "C"
                break

        if "tp" in ds:
            out_vars["precip"] = ds["tp"] * 1000.0  # m -> mm
            out_vars["precip"].attrs["units"] = "mm"

        u_name = next((n for n in ("10u", "u10") if n in ds), None)
        v_name = next((n for n in ("10v", "v10") if n in ds), None)
        if u_name and v_name:
            out_vars["wind10m"] = np.sqrt(ds[u_name] ** 2 + ds[v_name] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        for name in ("10fg", "gust", "fg10"):
            if name in ds:
                out_vars["gust10m"] = ds[name]
                out_vars["gust10m"].attrs["units"] = "m s-1"
                break

        out_ds = xr.Dataset(out_vars)
        if "time" not in out_ds.dims:
            dt_naive = pd.to_datetime(init_dt).tz_localize(None)
            out_ds = out_ds.expand_dims({"time": [dt_naive]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to process ECMWF data: {e}")
        return None
