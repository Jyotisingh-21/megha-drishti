import logging
import os
import shutil
from datetime import UTC, datetime

import pandas as pd
import xarray as xr

from nwpblend.harmonise.store import stack_models
from nwpblend.truth import fetch_real_truth

logger = logging.getLogger(__name__)


def verify_and_save_run(run_dt: datetime, config: dict, domain: dict) -> bool:
    """
    Attempts to pair an old run with truth data.
    Returns True if successfully verified and saved, False otherwise.
    """
    run_str = run_dt.strftime("%Y%m%d_%H%M")
    verified_path = os.path.join("data", "verified", f"pair_{run_str}.nc")

    # If already verified, we can safely return True (ready for cleanup)
    if os.path.exists(verified_path):
        return True

    archive_dir = "data/archive"
    expected_models = config.get("models", ["ecmwf_ifs", "gfs"])
    models = {}

    # Try to load raw files
    for m in expected_models:
        m_dir = os.path.join(archive_dir, m, run_str)
        raw_file = os.path.join(m_dir, "raw.grib")
        if os.path.exists(raw_file):
            try:
                if m == "ecmwf_ifs":
                    import nwpblend.ingest.ecmwf as ecmwf_ingest

                    ds = ecmwf_ingest._process_file(raw_file, domain, run_dt)
                elif m == "gfs":
                    import nwpblend.ingest.gfs as gfs_ingest

                    # GFS process file expects a list of files or a single file depending on implementation.
                    # We know gfs uses _process_files with a list of files!
                    ds = gfs_ingest._process_files([raw_file], domain, run_dt)
                else:
                    ds = None

                if ds is not None:
                    models[m] = ds
            except Exception as e:
                logger.error(f"Failed to load archived {m} for {run_str}: {e}")

    if not models:
        # No models found, nothing to verify. We can return True to allow cleanup of empty dirs.
        return True

    stacked = stack_models(models, list(models.keys()))

    # Fetch truth
    # The run covers from run_dt to run_dt + max_lead
    # max_lead = max(stacked.lead.values) hours
    max_lead_h = int(stacked.lead.max().item())
    end_dt = run_dt + pd.Timedelta(hours=max_lead_h)

    # Wait, fetching truth for a 10-day period
    truth = fetch_real_truth(run_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), domain)

    if truth is None:
        logger.info(f"Truth not fully available for {run_str} yet.")
        return False

    # Check if truth covers the end date
    # Usually truth is daily or hourly. Let's just check if the last required day is in truth.
    last_req_day = pd.Timestamp(end_dt).floor("D")
    truth_times = pd.to_datetime(truth.time.values)
    if len(truth_times) == 0 or truth_times.max() < last_req_day:
        logger.info(f"Truth data incomplete for {run_str} (needs up to {last_req_day.date()}).")
        return False

    # Save the pair
    os.makedirs(os.path.dirname(verified_path), exist_ok=True)
    # We can save it as a netcdf with groups or just merge them
    # To avoid naming conflicts, we can prefix truth variables
    truth_renamed = truth.rename({v: f"truth_{v}" for v in truth.data_vars})

    # We need to broadcast truth to the same time dimension?
    # Truth has 'time'. Stacked has 'time' (init) and 'lead'.
    # Truth valid time is init + lead.
    # We can just save them in the same file since they share 'lat' and 'lon'.
    try:
        combined = xr.merge([stacked, truth_renamed])
        combined.to_netcdf(verified_path)
        logger.info(f"Successfully saved verified pair to {verified_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save verified pair {run_str}: {e}")
        return False


def cleanup_archive(config, domain):
    """Deletes archived runs older than retention threshold ONLY IF verified."""
    retention_days = config.get("archive_retention_days", 14)
    cutoff = datetime.now(UTC) - pd.Timedelta(days=retention_days)

    archive_dir = "data/archive"
    if not os.path.exists(archive_dir):
        return

    for source in os.listdir(archive_dir):
        source_dir = os.path.join(archive_dir, source)
        if not os.path.isdir(source_dir):
            continue
        for run_dir in os.listdir(source_dir):
            try:
                run_dt = datetime.strptime(run_dir, "%Y%m%d_%H%M").replace(tzinfo=UTC)
                if run_dt < cutoff:
                    # Attempt verification
                    if verify_and_save_run(run_dt, config, domain):
                        path_to_delete = os.path.join(source_dir, run_dir)
                        shutil.rmtree(path_to_delete)
                        logger.info(f"Deleted old archive: {path_to_delete}")
                    else:
                        logger.info(f"Retaining {run_dir} because it is not yet verified.")
            except ValueError:
                pass
