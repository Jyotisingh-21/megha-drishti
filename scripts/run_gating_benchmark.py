import argparse
import logging
import os

import numpy as np
import pandas as pd
import shap
import torch
import xarray as xr

from nwpblend.blend.baselines import bma, equal_weight, ewa
from nwpblend.blend.explain import get_global_importance, plot_weight_maps
from nwpblend.blend.gating import GatingBlender, LightGBMBaseline
from nwpblend.regimes.cluster import RegimeClusterer
from nwpblend.verify.metrics import rmse


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        truth = xr.open_zarr("data/demo/truth.zarr").load()
        models = xr.open_zarr("data/processed/stacked_models.zarr").load()
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    times = models.time.values
    if len(times) < 60:
        logger.error("Dataset too small for gating train/test split.")
        return

    train_times = times[:60]
    test_times = times[60:]

    # 1. Cluster regimes
    clusterer = RegimeClusterer(n_clusters=4, random_state=42)
    clusterer.fit(truth.sel(time=train_times))

    # Get regime probs for the whole dataset
    len(models.lat) * len(models.lon)
    probs_list = []
    for t in times:
        _, p = clusterer.tag_day(truth.sel(time=[t]))
        probs_list.append(p)
    probs_array = np.concatenate(probs_list, axis=0)  # (time, 4)
    # Broadcast to spatial dims
    probs_array = probs_array[:, np.newaxis, np.newaxis, :]
    probs_array = np.broadcast_to(probs_array, (len(times), len(models.lat), len(models.lon), 4))

    regime_probs_da = xr.DataArray(
        probs_array,
        dims=["time", "lat", "lon", "cluster"],
        coords={"time": times, "lat": models.lat, "lon": models.lon, "cluster": range(4)},
    )

    # 2. Train GatingBlender for t2m
    logger.info("Training Gating Network for t2m...")
    models_train = models.sel(time=train_times)
    truth_train = truth.sel(time=train_times)
    avail_train = models.available.sel(time=train_times)

    blender = GatingBlender(n_models=len(models.model), is_precip=False)
    X_train, Y_train, M_train, F_train = blender.build_features(
        models_train, truth_train, avail_train, regime_probs_da, "t2m"
    )
    blender.fit(X_train, Y_train, M_train, F_train, epochs=20, lr=0.01)

    # 3. Test GatingBlender
    models_test = models.sel(time=test_times)
    truth_test = truth.sel(time=test_times)
    avail_test = models.available.sel(time=test_times)
    X_test, Y_test, M_test, F_test = blender.build_features(
        models_test, truth_test, avail_test, regime_probs_da, "t2m"
    )

    w_pred = blender.predict_weights(X_test, M_test)
    pred_gating = np.sum(w_pred * F_test.numpy(), axis=1)
    rmse_gating = np.sqrt(np.mean((pred_gating - Y_test.numpy()) ** 2))
    logger.info(f"Gating Network t2m RMSE: {rmse_gating:.4f}")

    # Baselines on test set (baselines need full dataset for history)
    bma_blend = bma(models, truth, models.available, window=30).sel(time=test_times)
    bma_rmse = float(rmse(bma_blend["t2m"].isel(lead=0), truth_test["t2m"]).mean().values)

    ewa_blend = ewa(models, truth, models.available, window=30).sel(time=test_times)
    ewa_rmse = float(rmse(ewa_blend["t2m"].isel(lead=0), truth_test["t2m"]).mean().values)

    logger.info(f"BMA t2m RMSE: {bma_rmse:.4f}")
    logger.info(f"EWA t2m RMSE: {ewa_rmse:.4f}")

    # SHAP and Maps
    # Convert w_pred back to xarray for a specific day
    # X_test covers (time * lead * lat * lon)
    # We'll just grab the first day in test set
    target_date = test_times[0]
    # Rebuild features for just that day to get exact indices
    X_d, _, M_d, _ = blender.build_features(
        models.sel(time=[target_date]),
        truth.sel(time=[target_date]),
        models.available.sel(time=[target_date]),
        regime_probs_da,
        "t2m",
    )
    w_d = blender.predict_weights(X_d, M_d)

    # Reshape w_d to (lead, lat, lon, model)
    w_d = w_d.reshape(len(models.lead), len(models.lat), len(models.lon), len(models.model))

    w_da = xr.DataArray(
        w_d,
        dims=["lead", "lat", "lon", "model"],
        coords={"lead": models.lead, "lat": models.lat, "lon": models.lon, "model": models.model},
    )
    # Add time dim back
    w_da = w_da.expand_dims({"time": [target_date]})
    w_ds = xr.Dataset({"weights": w_da})

    plot_weight_maps(w_ds, "t2m", str(target_date)[:10], str(models.lead.values[0]))

    # SHAP
    # Random sample of 100 points
    idx = np.random.choice(len(X_train), 100, replace=False)
    X_sample = X_train[idx]
    M_sample = M_train[idx]

    # Generate feature names
    names = ["sin_lat", "cos_lat", "sin_lon", "cos_lon", "sin_doy", "cos_doy", "lead_norm"]
    names += [f"regime_{i}" for i in range(4)]
    names += [f"fcst_{m}" for m in models.model.values]
    names += [f"mae_{m}" for m in models.model.values]

    # Compute
    try:
        importance = get_global_importance(blender.model, X_sample, M_sample, names)
        logger.info("Top 5 SHAP features:")
        for k, v in list(importance.items())[:5]:
            logger.info(f"  {k}: {v:.4f}")
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")


if __name__ == "__main__":
    main()
