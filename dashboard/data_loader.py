import os

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr


@st.cache_resource
def load_data():
    is_demo = True
    ds = None
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
