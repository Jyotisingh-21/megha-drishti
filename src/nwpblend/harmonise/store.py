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
    Supports varying 'lead' coordinates by taking an outer join.
    """
    valid_keys = [m for m in expected_models if m in model_datasets]
    if not valid_keys:
        raise ValueError("No models provided to stack.")

    # 1. Align all valid models to ensure consistent dimensions (e.g., missing leads filled with NaN)
    datasets_to_align = [model_datasets[m] for m in valid_keys]
    aligned_datasets = xr.align(*datasets_to_align, join="outer")
    aligned_dict = {m: ds for m, ds in zip(valid_keys, aligned_datasets)}

    # 2. Create empty dataset template for completely missing models based on the aligned shape
    template_ds = aligned_datasets[0].copy(deep=True)
    for v in template_ds.data_vars:
        template_ds[v] = xr.full_like(template_ds[v], np.nan, dtype=float)

    # 3. Assemble list of datasets in the exact order of expected_models
    aligned = []
    for m in expected_models:
        if m in aligned_dict:
            aligned.append(aligned_dict[m])
        else:
            aligned.append(template_ds.copy(deep=True))

    # 4. Concat safely
    stacked = xr.concat(aligned, pd.Index(expected_models, name="model"), join="outer")

    # 5. Compute available mask: (time, lead, model)
    # A model is available if it has at least some valid (non-NaN) data over spatial dims
    first_var = next(iter(stacked.data_vars.keys()))
    reduce_dims = [d for d in stacked[first_var].dims if d in ["lat", "lon"]]
    available = stacked[first_var].notnull().any(dim=reduce_dims)

    stacked["available"] = available
    return stacked
