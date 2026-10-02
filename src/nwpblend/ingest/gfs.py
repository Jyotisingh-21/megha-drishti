import concurrent.futures
import logging
import os
import shutil
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

logger = logging.getLogger(__name__)


def probe_and_fetch(target_date, domain, leads, variables):
    fs = s3fs.S3FileSystem(anon=True)

    if target_date == "latest":
        now = datetime.now(UTC)
        candidates = []
        for i in range(8):
            t = now - timedelta(hours=i * 6)
            h = (t.hour // 6) * 6
            candidates.append(t.replace(hour=h, minute=0, second=0, microsecond=0))
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)]

    success_ds = None
    chosen_run = None

    for cand in candidates:
        date_str = cand.strftime("%Y%m%d")
        time_str = f"{cand.hour:02d}"

        archive_dir = f"data/archive/gfs/{cand.strftime('%Y%m%d_%H%M')}"
        os.makedirs(archive_dir, exist_ok=True)

        missing_leads = False
        target_files = []

        # GFS variable map (exact cfgrib / index strings)
        gfs_vars = []
        if "t2m" in variables:
            gfs_vars.append(":TMP:2 m above ground:")
        if "precip" in variables:
            gfs_vars.append(":PRATE:surface:")
        if "wind10m" in variables:
            gfs_vars.append(":UGRD:10 m above ground:")
            gfs_vars.append(":VGRD:10 m above ground:")
        if "gust10m" in variables:
            gfs_vars.append(":GUST:surface:")

        for lead in leads:
            lead_str = f"{lead:03d}"
            target_file = os.path.join(archive_dir, f"raw_f{lead_str}.grib")
            target_files.append(target_file)

            if not os.path.exists(target_file):
                s3_path = f"noaa-gfs-bdp-pds/gfs.{date_str}/{time_str}/atmos/gfs.t{time_str}z.pgrb2.0p25.f{lead_str}"
                idx_path = s3_path + ".idx"

                if fs.exists(idx_path):
                    logger.info(f"Downloading GFS run {cand} lead {lead} via index...")
                    try:

                        def do_fetch(idx_p, vars_list, s3_p, out_file):
                            idx_data = fs.cat(idx_p).decode("utf-8").splitlines()
                            ranges = []
                            for i, line in enumerate(idx_data):
                                for gv in vars_list:
                                    if gv in line:
                                        start = int(line.split(":")[1])
                                        end = None
                                        if i + 1 < len(idx_data):
                                            end = int(idx_data[i + 1].split(":")[1]) - 1
                                        ranges.append((start, end))

                            if not ranges:
                                raise ValueError("No valid variables found in index")

                            with (
                                fs.open(s3_p, "rb", fill_cache=False) as f_in,
                                open(out_file, "wb") as f_out,
                            ):
                                for start, end in ranges:
                                    f_in.seek(start)
                                    length = end - start + 1 if end else 2000000
                                    f_out.write(f_in.read(length))

                        import yaml

                        try:
                            with open("configs/default.yaml", "r") as f:
                                config = yaml.safe_load(f)
                        except Exception:
                            config = {}
                        download_timeout = config.get("download_timeout", 600)

                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(do_fetch, idx_path, gfs_vars, s3_path, target_file)
                        future.result(timeout=download_timeout)
                    except Exception as e:
                        if isinstance(e, concurrent.futures.TimeoutError):
                            logger.error(
                                f"GFS download timed out after {download_timeout}s for lead {lead}"
                            )
                        else:
                            logger.error(f"GFS download failed: {e}")
                        missing_leads = True
                        break
                else:
                    logger.info(f"GFS run {cand} lead {lead} missing on AWS.")
                    missing_leads = True
                    break

        if not missing_leads:
            logger.info(f"Found complete GFS run {cand}")
            success_ds = _process_files(target_files, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.error(f"Failed to process downloaded files for {cand}")

    if chosen_run:
        logger.info(f"Selected GFS run: {chosen_run}")
        return success_ds, chosen_run
    else:
        logger.error("No valid GFS runs found in probing window.")
        return None, None


def _process_files(target_files, domain, init_dt):
    datasets = []

    for target_file in target_files:
        try:
            ds = xr.open_dataset(
                target_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface"}
            )
            ds2 = xr.open_dataset(
                target_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "heightAboveGround"}
            )

            ds = xr.merge([ds, ds2], compat="override")

            lat_slice = slice(domain["lat_max"] + 1, domain["lat_min"] - 1)
            lon_slice = slice(domain["lon_min"] - 1, domain["lon_max"] + 1)
            if ds.latitude.values[0] < ds.latitude.values[-1]:
                lat_slice = slice(domain["lat_min"] - 1, domain["lat_max"] + 1)

            ds_subset = ds.sel(latitude=lat_slice, longitude=lon_slice)

            new_lats = np.arange(domain["lat_max"], domain["lat_min"] - 0.01, -domain["resolution"])
            new_lons = np.arange(domain["lon_min"], domain["lon_max"] + 0.01, domain["resolution"])
            ds_subset = ds_subset.interp(latitude=new_lats, longitude=new_lons, method="linear")

            datasets.append(ds_subset)
        except Exception as e:
            logger.error(f"Failed to process GFS file {target_file}: {e}")
            return None

    if not datasets:
        return None

    try:
        combined = xr.concat(datasets, dim="step")
        combined = combined.rename({"latitude": "lat", "longitude": "lon"})
        if "step" in combined.coords:
            combined = combined.rename({"step": "lead"})

        out_vars = {}
        if "t2m" in combined:
            out_vars["t2m"] = combined["t2m"] - 273.15
            out_vars["t2m"].attrs["units"] = "C"

        if "prate" in combined:
            out_vars["precip"] = combined["prate"] * 86400.0
            out_vars["precip"].attrs["units"] = "mm"
        elif "tp" in combined:
            out_vars["precip"] = combined["tp"]
            out_vars["precip"].attrs["units"] = "mm"

        if "u10" in combined and "v10" in combined:
            out_vars["wind10m"] = np.sqrt(combined["u10"] ** 2 + combined["v10"] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"
        elif "10u" in combined and "10v" in combined:
            out_vars["wind10m"] = np.sqrt(combined["10u"] ** 2 + combined["10v"] ** 2)
            out_vars["wind10m"].attrs["units"] = "m s-1"

        if "gust" in combined:
            out_vars["gust10m"] = combined["gust"]
            out_vars["gust10m"].attrs["units"] = "m s-1"
        elif "10fg" in combined:
            out_vars["gust10m"] = combined["10fg"]
            out_vars["gust10m"].attrs["units"] = "m s-1"

        out_ds = xr.Dataset(out_vars)
        if "time" not in out_ds.coords:
            out_ds = out_ds.expand_dims({"time": [init_dt]})

        return out_ds
    except Exception as e:
        logger.error(f"Failed to combine GFS data: {e}")
        return None
