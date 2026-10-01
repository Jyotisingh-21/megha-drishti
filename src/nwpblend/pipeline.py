import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from nwpblend.harmonise.regrid import regrid
from nwpblend.harmonise.store import stack_models, write_store

# Adjust imports based on the actual modules created
from nwpblend.ingest import aifs, ecmwf, gfs, ncmrwf_adapter, truth

logger = logging.getLogger(__name__)


def build_dataset(date_range: list[str], expected_models: list[str], truth_source: str = "era5"):
    """
    Ties ingest, regrid, and store together.
    """
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    domain = config["domain"]
    leads = config["lead_times_days"]
    variables = config["variables"]

    # Target grid
    res = domain["resolution"]
    lats = np.arange(domain["lat_max"], domain["lat_min"] - res / 2, -res)
    lons = np.arange(domain["lon_min"], domain["lon_max"] + res / 2, res)
    target_grid = xr.Dataset(coords={"lat": lats, "lon": lons})

    # Store path
    out_path = "data/processed/stacked_models.zarr"

    for date in date_range:
        logger.info(f"Processing {date}...")

        # 1. Ingest
        model_ds = {}
        for m in expected_models:
            if m == "ecmwf_ifs":
                ds = ecmwf.fetch(date, domain, leads, variables)
            elif m == "gfs":
                ds = gfs.fetch(date, domain, leads, variables)
            elif m == "aifs":
                ds = aifs.fetch(date, domain, leads, variables)
            elif m == "ncum_g":
                ds = ncmrwf_adapter.fetch(date, domain, leads, variables)
            else:
                ds = None

            if ds is not None:
                # 2. Regrid
                model_ds[m] = regrid(ds, target_grid)

        if not model_ds:
            logger.warning(f"No models available for {date}.")
            continue

        # 3. Stack
        stacked = stack_models(model_ds, expected_models)

        # 4. Store
        write_store(stacked, out_path, append=True)

    logger.info("Pipeline build_dataset complete.")
