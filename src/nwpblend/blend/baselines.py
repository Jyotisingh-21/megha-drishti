from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm


def equal_weight(models: xr.Dataset, available: xr.DataArray) -> xr.Dataset:
    """
    Simple unweighted mean of all available models.
    """
    # Create mask where available is True
    # models: dims (time, lead, lat, lon, model)
    # available: dims (time, model)

    # Broadcast available to full dataset shape
    mask = available.broadcast_like(models)

    # Mask out missing models with NaN
    masked_models = models.where(mask)

    # Mean over model dimension (nanmean)
    return masked_models.mean(dim="model")


def ewa(
    models: xr.Dataset,
    truth: xr.Dataset,
    available: xr.DataArray,
    window: int = 30,
    eta: float = 0.1,
) -> xr.Dataset:
    """
    Exponentially Weighted Averaging based on recent MAE.
    Returns the blended forecast for the dates in `models`.
    Requires `truth` to contain prior days for trailing calculation.
    """
    blended = models.isel(model=0).copy() * np.nan
    mask = available.broadcast_like(models)
    masked_models = models.where(mask)

    for t_idx, t in enumerate(models.time.values):
        t_dt = pd.to_datetime(t)
        # Trailing window [t - window days, t - 1 day]
        t_start = t_dt - pd.Timedelta(days=window)
        t_end = t_dt - pd.Timedelta(days=1)

        # We need historical models and truth.
        # In a real streaming system, we maintain rolling errors.
        # Here we slice the provided datasets.
        try:
            hist_models = models.sel(time=slice(t_start, t_end))
            hist_truth = truth.sel(time=slice(t_start, t_end))
            available.sel(time=slice(t_start, t_end))
        except KeyError:
            # Not enough history in provided arrays, fall back to equal weight
            weights = xr.ones_like(available.sel(time=t), dtype=float)
            weights = weights.where(available.sel(time=t), 0.0)
            weights = weights / weights.sum(dim="model")

            for var in models.data_vars:
                if var == "available":
                    continue
                blended[var].loc[{"time": t}] = (
                    (masked_models[var].sel(time=t) * weights).sum(dim="model", skipna=True).values
                )
            continue

        if len(hist_models.time) == 0:
            # Fallback to equal weight
            weights = xr.ones_like(available.sel(time=t), dtype=float)
            weights = weights.where(available.sel(time=t), 0.0)
            weights = weights / weights.sum(dim="model")

            for var in models.data_vars:
                if var == "available":
                    continue
                if var == "available":
                    continue

                val = (
                    (masked_models[var].sel(time=t) * weights).sum(dim="model", skipna=True).values
                )
                blended[var].loc[{"time": t}] = val
            continue

        # Calculate MAE per model over the history
        # hist_models: (time, lead, lat, lon, model)
        # hist_truth: (time, lat, lon)
        # Broadcast truth to match leads

        for var in models.data_vars:
            if var == "available":
                continue
            err = np.abs(hist_models[var] - hist_truth[var])
            mae = err.mean(dim="time")  # (lead, lat, lon, model)

            # weights ∝ exp(-eta * MAE)
            w = np.exp(-eta * mae)

            # Apply current day availability mask
            curr_avail = available.sel(time=t)
            w = w.where(curr_avail, 0.0)

            # Renormalize
            w = w / w.sum(dim="model")

            # Apply
            blended[var].loc[{"time": t}] = (
                (masked_models[var].sel(time=t) * w).sum(dim="model", skipna=True).values
            )

    return blended


def bma(
    models: xr.Dataset, truth: xr.Dataset, available: xr.DataArray, window: int = 30
) -> xr.Dataset:
    """
    Bayesian Model Averaging.
    For demonstration on synthetic data, we approximate EM over the sliding window.
    Weights are optimized to minimize MSE (proxy for Gaussian BMA weights).
    """
    blended = models.isel(model=0).copy() * np.nan
    mask = available.broadcast_like(models)
    masked_models = models.where(mask)

    # We will compute a simple inverse-variance weighting as a fast BMA approximation
    # to avoid solving EM iteratively at every single grid point which takes hours.
    # W_k \propto 1 / Variance(Error_k)

    for t_idx, t in enumerate(models.time.values):
        t_dt = pd.to_datetime(t)
        t_start = t_dt - pd.Timedelta(days=window)
        t_end = t_dt - pd.Timedelta(days=1)

        try:
            hist_models = models.sel(time=slice(t_start, t_end))
            hist_truth = truth.sel(time=slice(t_start, t_end))
        except KeyError:
            weights = xr.ones_like(available.sel(time=t), dtype=float)
            weights = weights.where(available.sel(time=t), 0.0)
            weights = weights / weights.sum(dim="model")
            for var in models.data_vars:
                if var == "available":
                    continue
                blended[var].loc[{"time": t}] = (
                    (masked_models[var].sel(time=t) * weights).sum(dim="model", skipna=True).values
                )
            continue

        if len(hist_models.time) < 2:
            weights = xr.ones_like(available.sel(time=t), dtype=float)
            weights = weights.where(available.sel(time=t), 0.0)
            weights = weights / weights.sum(dim="model")
            for var in models.data_vars:
                if var == "available":
                    continue
                blended[var].loc[{"time": t}] = (
                    (masked_models[var].sel(time=t) * weights).sum(dim="model", skipna=True).values
                )
            continue

        for var in models.data_vars:
            if var == "available":
                continue
            err = hist_models[var] - hist_truth[var]
            mse = (err**2).mean(dim="time")

            # W \propto 1 / (MSE + eps)
            w = 1.0 / (mse + 1e-6)

            curr_avail = available.sel(time=t)
            w = w.where(curr_avail, 0.0)
            w = w / w.sum(dim="model")

            blended[var].loc[{"time": t}] = (
                (masked_models[var].sel(time=t) * w).sum(dim="model", skipna=True).values
            )

    return blended
