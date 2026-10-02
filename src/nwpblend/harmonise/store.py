import os

# no typing import needed
import numpy as np
import pandas as pd
import xarray as xr


def write_store(ds: xr.Dataset, path: str, append: bool = False):
    """
    Write or append Dataset to a Zarr store with standard chunking.
    Chunking: time=1, lead=all (-1), lat=64, lon=64, model=-1.
    """
    chunks = {}
    if "time" in ds.coords:
        chunks["time"] = 1
    if "lead" in ds.coords:
        chunks["lead"] = -1
    if "model" in ds.coords:
        chunks["model"] = -1
    if "lat" in ds.coords:
        chunks["lat"] = min(64, len(ds.lat))
    if "lon" in ds.coords:
        chunks["lon"] = min(64, len(ds.lon))

    ds_chunked = ds.chunk(chunks)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if append and os.path.exists(path):
        ds_chunked.to_zarr(path, mode="a", append_dim="time")
    else:
        ds_chunked.to_zarr(path, mode="w")


def stack_models(model_datasets: dict[str, xr.Dataset], expected_models: list[str]) -> xr.Dataset:
    """
    Stacks a dict of model datasets into a single Dataset with a 'model' dimension.
    Fills missing models with NaNs and computes a boolean 'available' mask.
    """
    aligned = []

    # Create empty dataset template for missing models
    template_ds = None
    for m in expected_models:
        if m in model_datasets:
            template_ds = model_datasets[m].copy(deep=True)
            for v in template_ds.data_vars:
                template_ds[v] = xr.full_like(template_ds[v], np.nan, dtype=float)
            break

    if template_ds is None:
        raise ValueError("No models provided to stack.")

    for m in expected_models:
        if m in model_datasets:
            ds = model_datasets[m]
        else:
            ds = template_ds.copy(deep=True)
        aligned.append(ds)

    stacked = xr.concat(aligned, pd.Index(expected_models, name="model"), join="override")

    # Compute available mask: (time, model)
    # A model is available if it has at least some valid (non-NaN) data
    # We use a proxy variable like t2m or precip.
    first_var = next(iter(stacked.data_vars.keys()))

    # Has valid data over lat/lon/lead
    reduce_dims = [d for d in stacked[first_var].dims if d not in ["time", "model"]]
    available = stacked[first_var].notnull().any(dim=reduce_dims)

    stacked["available"] = available
    return stacked
