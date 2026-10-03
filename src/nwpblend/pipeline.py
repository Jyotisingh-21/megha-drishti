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


def verify_and_save_run(run_dt: datetime, config: dict, domain: dict) -> bool:
    run_str = run_dt.strftime("%Y%m%d_%H%M")
    verified_path = os.path.join("data", "verified", f"pair_{run_str}.nc")

    if os.path.exists(verified_path):
        return True

    archive_dir = "data/archive"
    expected_models = config.get("models", ["ecmwf_ifs", "gfs"])
    models = {}

    for m in expected_models:
        m_dir = os.path.join(archive_dir, m, run_str)
        if not os.path.exists(m_dir):
            continue

        raw_file = os.path.join(m_dir, "raw.grib")
        if os.path.exists(raw_file):
            try:
                if m == "ecmwf_ifs":
                    import nwpblend.ingest.ecmwf as ecmwf_ingest

                    ds = ecmwf_ingest._process_file(raw_file, domain, run_dt)
                elif m == "gfs":
                    import nwpblend.ingest.gfs as gfs_ingest

                    # _process_files expects a list of files
                    ds = gfs_ingest._process_files([raw_file], domain, run_dt)
                else:
                    ds = None

                if ds is not None:
                    # Strip any timezone information
                    for c in ds.coords:
                        if (
                            hasattr(ds[c].values, "dtype")
                            and "datetime64" in str(ds[c].values.dtype)
                            and "UTC" in str(ds[c].values.dtype)
                        ):
                            ds = ds.assign_coords(
                                {c: pd.to_datetime(ds[c].values).tz_localize(None)}
                            )
                    models[m] = ds
            except Exception as e:
                logger.error(f"Failed to load archived {m} for {run_str}: {e}")

    if not models:
        return True

    from nwpblend.harmonise.store import stack_models

    stacked = stack_models(models, list(models.keys()))

    max_lead_h = int(stacked.lead.max().item())
    end_dt = run_dt + pd.Timedelta(hours=max_lead_h)

    from nwpblend.truth import fetch_real_truth

    truth = fetch_real_truth(run_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), domain)

    if truth is None:
        logger.info(f"Truth not fully available for {run_str} yet.")
        return False

    last_req_day = pd.Timestamp(end_dt).tz_localize(None).floor("D")
    truth_times = pd.to_datetime(truth.time.values)
    if truth_times.tz is not None:
        truth_times = truth_times.tz_localize(None)

    if len(truth_times) == 0 or truth_times.max() < last_req_day:
        logger.info(f"Truth data incomplete for {run_str} (needs up to {last_req_day.date()}).")
        return False

    try:
        truth_renamed = truth.rename({v: f"truth_{v}" for v in truth.data_vars})

        pair_dir = verified_path.replace(".nc", "")
        os.makedirs(pair_dir, exist_ok=True)
        stacked.to_zarr(os.path.join(pair_dir, "models.zarr"), mode="w")
        truth_renamed.to_zarr(os.path.join(pair_dir, "truth.zarr"), mode="w")

        logger.info(f"Successfully saved verified pair to {pair_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to save verified pair {run_str}: {e}")
        return False


def cleanup_archive(config, domain):
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
                    if verify_and_save_run(run_dt, config, domain):
                        path_to_delete = os.path.join(source_dir, run_dir)
                        shutil.rmtree(path_to_delete)
                        logger.info(f"Deleted old archive: {path_to_delete}")
                    else:
                        logger.info(f"Retaining {run_dir} because it is not yet verified.")
            except ValueError:
                pass


def check_lock():
    lockfile = "data/logs/pipeline.lock"
    if os.path.exists(lockfile):
        with open(lockfile, "r") as f:
            try:
                data = json.load(f)
                start_time = datetime.fromisoformat(data["start_time"])
                pid = data.get("pid")

                # Check if process is actually running
                is_running = False
                if pid:
                    import psutil

                    try:
                        p = psutil.Process(pid)
                        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                            is_running = True
                    except psutil.NoSuchProcess:
                        pass

                # Configurable limit: 4 hours
                if not is_running:
                    logger.warning(f"Lock found but PID {pid} is not running. Overriding.")
                    os.remove(lockfile)
                elif (datetime.now(UTC) - start_time).total_seconds() > 4 * 3600:
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


def run_daily(
    date: str,
    domain: dict,
    demo: bool = False,
    skip_download: bool = False,
    quick: bool = False,
    max_leads: int | None = None,
):
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
        if quick:
            leads = [24, 48, 72]
        if max_leads:
            leads = leads[:max_leads]

        cleanup_archive(config, domain)

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
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(
                                f"ECMWF partial run ({len(ds.lead)}/{len(leads)} leads)."
                            )
                    else:
                        report["warnings"].append("ECMWF failed. Dropping from blend.")

                # Fetch GFS
                if "gfs" in expected_models:
                    ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["gfs"] = ds
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(
                                f"GFS partial run ({len(ds.lead)}/{len(leads)} leads)."
                            )
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
                import glob

                blender_used = "equal_weight"
                blended = None

                # Setup available mask for current run
                available = xr.DataArray(
                    [True] * len(stacked_models.model),
                    coords={"model": stacked_models.model},
                    dims=["model"],
                ).expand_dims(time=stacked_models.time)

                if demo:
                    from nwpblend.blend.baselines import equal_weight

                    blended = equal_weight(stacked_models, available)
                else:
                    # Look for historical verified data
                    hist_models_paths = sorted(glob.glob("data/verified/pair_*/models.zarr"))
                    hist_truth_paths = sorted(glob.glob("data/verified/pair_*/truth.zarr"))

                    min_history_days = config.get("min_history_days", 30)
                    hist_pairs = len(hist_models_paths)

                    if hist_pairs >= min_history_days:
                        # TODO: train bias correction and gating network
                        # For now, fallback to BMA or EWA
                        pass

                    if hist_pairs >= 30:
                        from nwpblend.blend.baselines import ewa

                        try:
                            # We load history lazily
                            h_m = xr.open_mfdataset(
                                hist_models_paths,
                                engine="zarr",
                                concat_dim="time",
                                combine="nested",
                            )
                            h_t = xr.open_mfdataset(
                                hist_truth_paths, engine="zarr", concat_dim="time", combine="nested"
                            )
                            h_t = h_t.rename({f"truth_{v}": v for v in h_t.data_vars})

                            # EWA requires hist truth for past days
                            # Just concat the history and current to satisfy ewa expectations
                            comb_models = xr.concat([h_m, stacked_models], dim="time")
                            comb_avail = xr.DataArray(
                                True, coords={"time": comb_models.time, "model": comb_models.model}
                            )

                            blended_all = ewa(comb_models, h_t, comb_avail, window=30)
                            blended = blended_all.isel(time=[-1])  # get current
                            blender_used = "ewa"
                        except Exception as e:
                            logger.warning(f"EWA failed: {e}. Falling back to equal_weight.")

                    if blended is None:
                        from nwpblend.blend.baselines import equal_weight

                        blended = equal_weight(stacked_models, available)
                        blender_used = "equal_weight"

                metadata = {
                    "issue_date": date,
                    "models_used": list(stacked_models.model.values),
                    "blender": blender_used,
                }
                export_netcdf(blended, out_file, metadata=metadata)

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
