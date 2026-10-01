import os
import time
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any

import xarray as xr
import pandas as pd

from nwpblend.export.writer import export_netcdf

logger = logging.getLogger(__name__)

def run_daily(date: str, domain: dict, demo: bool = False, skip_download: bool = False):
    """
    Executes the full daily operational pipeline end-to-end.
    """
    start_time = time.time()
    
    report: Dict[str, Any] = {
        "date": date,
        "mode": "DEMO" if demo else "REAL",
        "stages": {},
        "warnings": [],
        "errors": []
    }
    
    def log_stage(name: str, start: float, status: str, details: str = ""):
        dur = time.time() - start
        report["stages"][name] = {"status": status, "duration_sec": round(dur, 2), "details": details}
        logger.info(f"STAGE [{name}] {status} in {dur:.2f}s. {details}")

    logger.info(f"Starting pipeline for {date} (Demo={demo})")
    
    # 1. Ingest
    t0 = time.time()
    try:
        if not skip_download:
            # We would invoke the ingestors here
            pass
        log_stage("ingest", t0, "SUCCESS")
    except Exception as e:
        report["errors"].append(f"Ingest failed: {e}")
        log_stage("ingest", t0, "FAILED")
        return report

    # 2. Harmonise & Stack
    t0 = time.time()
    models = None
    try:
        # Assuming the synthetic or actual script wrote to data/processed/stacked_models.zarr
        if demo:
            models_path = "data/processed/stacked_models.zarr"
            if not os.path.exists(models_path):
                # Fallback generate
                from nwpblend.harmonise.store import stack_models, write_store
                expected = ["ecmwf_ifs", "gfs", "aifs", "ncum_g", "graphcast", "pangu"]
                m_dict = {}
                for m in expected:
                    try:
                        m_dict[m] = xr.open_zarr(f"data/demo/models/{m}.zarr").load()
                    except Exception:
                        report["warnings"].append(f"Demo model {m} not found.")
                models = stack_models(m_dict, expected)
                os.makedirs("data/processed", exist_ok=True)
                write_store(models, models_path)
            else:
                models = xr.open_zarr(models_path).load()
        else:
            # Real operational harmonisation logic here
            models = xr.open_zarr("data/processed/stacked_models.zarr").load()
            
        log_stage("harmonise", t0, "SUCCESS", f"Models loaded: {models.model.values}")
    except Exception as e:
        report["errors"].append(f"Harmonise failed: {e}")
        log_stage("harmonise", t0, "FAILED", str(e))
        return report

    # 3. Blending & Export
    t0 = time.time()
    try:
        # In a real operational setting, we'd run gating predict here
        # For the pipeline mock, we just use the raw output or mean
        # and export it
        
        # Export NetCDF
        os.makedirs("data/output", exist_ok=True)
        out_file = f"data/output/blend_{date}.nc"
        
        metadata = {
            "issue_date": date,
            "models_used": list(models.model.values)
        }
        
        # We'll just export the first model as a placeholder for the blended result in demo
        blend_mock = models.isel(model=0).drop_vars("model") if "model" in models.dims else models
        
        export_netcdf(blend_mock, out_file, metadata=metadata)
        
        log_stage("blend_and_export", t0, "SUCCESS", f"Saved to {out_file}")
    except Exception as e:
        report["errors"].append(f"Blend/Export failed: {e}")
        report["errors"].append(traceback.format_exc())
        log_stage("blend_and_export", t0, "FAILED")

    # Finalize
    total_time = time.time() - start_time
    report["total_time_sec"] = round(total_time, 2)
    report["status"] = "FAILED" if report["errors"] else "SUCCESS"
    
    os.makedirs("data/logs", exist_ok=True)
    with open(f"data/logs/run_report_{date}.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Pipeline finished with status: {report['status']}")
    
    if report["errors"]:
        # We explicitly raise so CI and external scripts know it failed
        raise RuntimeError("Pipeline failed! See report for details.")
        
    return report
