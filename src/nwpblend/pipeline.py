import json
import logging
import os
import shutil
import time
import traceback
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from nwpblend.export.writer import export_netcdf

logger = logging.getLogger(__name__)


def cleanup_archive(config):
    """Deletes archived runs older than retention threshold."""
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
                    path_to_delete = os.path.join(source_dir, run_dir)
                    shutil.rmtree(path_to_delete)
                    logger.info(f"Deleted old archive: {path_to_delete}")
            except ValueError:
                pass


def check_lock():
    lockfile = "data/logs/pipeline.lock"
    if os.path.exists(lockfile):
        with open(lockfile, "r") as f:
            try:
                data = json.load(f)
                start_time = datetime.fromisoformat(data["start_time"])
                # Configurable limit: 4 hours
                if (datetime.now(UTC) - start_time).total_seconds() > 4 * 3600:
                    logger.warning(f"Stale lock found from {start_time}. Overriding.")
                    os.remove(lockfile)
                else:
                    return False
            except Exception:
                os.remove(lockfile)

    os.makedirs(os.path.dirname(lockfile), exist_ok=True)
    with open(lockfile, "w") as f:
        json.dump({"pid": os.getpid(), "start_time": datetime.now(UTC).isoformat()}, f)
    return True


def release_lock():
    lockfile = "data/logs/pipeline.lock"
    if os.path.exists(lockfile):
        os.remove(lockfile)


def run_daily(date: str, domain: dict, demo: bool = False, skip_download: bool = False):
    """
    Executes the full daily operational pipeline end-to-end.
    """
    if not check_lock():
        logger.error("Pipeline is already running. Exiting.")
        return {"status": "SKIPPED", "errors": ["Locked"]}

    start_time = time.time()

    report: dict[str, Any] = {
        "date": date,
        "mode": "DEMO" if demo else "REAL",
        "stages": {},
        "warnings": [],
        "errors": [],
    }

    def log_stage(name: str, start: float, status: str, details: str = ""):
        dur = time.time() - start
        report["stages"][name] = {
            "status": status,
            "duration_sec": round(dur, 2),
            "details": details,
        }
        logger.info(f"STAGE [{name}] {status} in {dur:.2f}s. {details}")

    logger.info(f"Starting pipeline for {date} (Demo={demo})")

    try:
        # Load config
        with open("configs/default.yaml", "r") as f:
            config = yaml.safe_load(f)

        variables = config.get("variables", ["precip", "t2m", "wind10m", "gust10m"])
        leads = [i * 24 for i in config.get("lead_times_days", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]

        cleanup_archive(config)

        # 1. Ingest
        t0 = time.time()
        models = {}
        expected_models = config.get("models", ["ecmwf_ifs", "gfs"])

        try:
            if demo:
                # In demo mode, just load the static demo files
                for m in expected_models:
                    try:
                        models[m] = xr.open_zarr(f"data/demo/models/{m}.zarr").load()
                    except Exception:
                        pass
                log_stage("ingest", t0, "SUCCESS", "Loaded demo data")
            elif not skip_download:
                import nwpblend.ingest.ecmwf as ecmwf_ingest
                import nwpblend.ingest.gfs as gfs_ingest

                # Fetch ECMWF
                if "ecmwf_ifs" in expected_models:
                    ds, _run_dt = ecmwf_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["ecmwf_ifs"] = ds
                    else:
                        report["warnings"].append("ECMWF failed. Dropping from blend.")

                # Fetch GFS
                if "gfs" in expected_models:
                    ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["gfs"] = ds
                    else:
                        report["warnings"].append("GFS failed. Dropping from blend.")

                log_stage("ingest", t0, "SUCCESS")
        except Exception as e:
            report["errors"].append(f"Ingest failed: {e}")
            log_stage("ingest", t0, "FAILED")

        # 2. Harmonise & Stack
        t0 = time.time()
        stacked_models = None
        try:
            if not models:
                raise ValueError("No models available to stack.")

            from nwpblend.harmonise.store import stack_models, write_store

            # Use only available models and normalize their weights logically later
            stacked_models = stack_models(models, list(models.keys()))
            os.makedirs("data/processed", exist_ok=True)
            models_path = "data/processed/stacked_models.zarr"
            write_store(stacked_models, models_path)

            log_stage("harmonise", t0, "SUCCESS", f"Models loaded: {stacked_models.model.values}")
        except Exception as e:
            import traceback

            report["errors"].append(f"Harmonise failed: {e}\n{traceback.format_exc()}")
            log_stage("harmonise", t0, "FAILED", str(e))

        # 3. Blending & Export
        t0 = time.time()
        try:
            if stacked_models is not None:
                os.makedirs("data/output", exist_ok=True)
                out_file = f"data/output/blend_{date}.nc"
                metadata = {"issue_date": date, "models_used": list(stacked_models.model.values)}
                blend_mock = (
                    stacked_models.isel(model=0).drop_vars("model")
                    if "model" in stacked_models.dims
                    else stacked_models
                )
                export_netcdf(blend_mock, out_file, metadata=metadata)

                # Keep a history of issued forecasts
                dt_obj = pd.to_datetime(stacked_models.time.values[0])
                archive_out = f"data/output/archive/blend_{dt_obj.strftime('%Y%m%d')}.nc"
                os.makedirs(os.path.dirname(archive_out), exist_ok=True)
                import shutil

                shutil.copy2(out_file, archive_out)
                log_stage("blend_and_export", t0, "SUCCESS", f"Saved to {out_file}")
            else:
                log_stage("blend_and_export", t0, "SKIPPED")
        except Exception as e:
            report["errors"].append(f"Blend/Export failed: {e}")
            report["errors"].append(traceback.format_exc())
            log_stage("blend_and_export", t0, "FAILED")

    finally:
        # Finalize
        total_time = time.time() - start_time
        report["total_time_sec"] = round(total_time, 2)
        report["status"] = "FAILED" if report["errors"] else "SUCCESS"

        os.makedirs("data/logs", exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        with open(f"data/logs/run_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2)

        # For dashboard
        with open("data/logs/run_report_latest.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pipeline finished with status: {report['status']}")
        release_lock()

    if report["errors"]:
        raise RuntimeError("Pipeline failed! See report for details.")

    return report
