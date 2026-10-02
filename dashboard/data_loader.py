import glob
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr


@st.cache_data(ttl=3600)
def _load_data_cached(_dummy_time):
    # Dummy argument forces reload when time changes if we want, but ttl=3600 handles it
    is_demo = True
    ds = None
    
    # Check archive for latest run metadata
    archives = sorted(glob.glob("data/output/archive/blend_*.nc"))
    if archives:
        try:
            # Check the attributes of the latest archive
            latest_ds = xr.open_dataset(archives[-1], engine="netcdf4")
            if "demo" in latest_ds.attrs:
                # NetCDF attrs might be string 'True' or 'False', or integer 0/1
                val = latest_ds.attrs["demo"]
                is_demo = str(val).lower() in ("true", "1")
            latest_ds.close()
        except Exception:
            pass

    if os.path.exists("data/processed/stacked_models.zarr"):
        try:
            ds = xr.open_zarr("data/processed/stacked_models.zarr").load()
        except Exception:
            pass

    if ds is None:
        try:
            ds = xr.open_zarr("data/demo/models/ecmwf_ifs.zarr").load()
        except Exception:
            pass

    return ds, is_demo

def load_data():
    return _load_data_cached(time.time() // 3600)
