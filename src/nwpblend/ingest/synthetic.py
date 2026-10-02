import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

def _smooth(x, sigma=2.0):
    # If 3D (time, lat, lon), smooth over last two axes
    # If 4D (time, lead, lat, lon), smooth over last two axes
    if x.ndim >= 2:
        return gaussian_filter(x, sigma=sigma, axes=(-2, -1))
    return x

def generate_synthetic_data(
    lat_min=6.0, lat_max=37.0, lon_min=68.0, lon_max=98.0, resolution=0.5, days=90
):
    """
    Generate synthetic truth, deterministic models, and ensemble models for the demo.
    """
    np.random.seed(42)

    lats = np.arange(lat_min, lat_max + resolution / 2, resolution)
    lons = np.arange(lon_min, lon_max + resolution / 2, resolution)

    # 90 days of initializations
    times = pd.date_range("2024-06-01", periods=days, freq="D")
    leads = np.arange(24, 24 * 10 + 24, 24)

    regime_labels = np.random.choice([0, 1, 2, 3], size=days, p=[0.4, 0.3, 0.1, 0.2])
    regimes_da = xr.DataArray(regime_labels, dims=["time"], coords={"time": times}, name="regime")
    regimes_ds = xr.Dataset({"regime": regimes_da})

    lon2d, lat2d = np.meshgrid(lons, lats)

    # Deterministic spatial masks
    # Western Ghats proxy
    is_ghats = ((lon2d >= 73) & (lon2d <= 77) & (lat2d >= 10) & (lat2d <= 20)).astype(float)
    # Himalayas
    is_himalayas = ((lat2d >= 28) & (lat2d <= 36) & (lon2d >= 74) & (lon2d <= 96)).astype(float)
    # IGP / Rajasthan (hot summer)
    is_hot = ((lat2d >= 24) & (lat2d <= 30) & (lon2d >= 70) & (lon2d <= 82)).astype(float)
    # Northeast rain
    is_ne = ((lat2d >= 24) & (lat2d <= 28) & (lon2d >= 90) & (lon2d <= 98)).astype(float)

    is_ghats = _smooth(is_ghats, 1.5)
    is_himalayas = _smooth(is_himalayas, 1.5)
    is_hot = _smooth(is_hot, 2.0)
    is_ne = _smooth(is_ne, 1.5)

    # 1. TRUTH GENERATION
    # Base precip: smooth noise
    precip_noise = np.random.gamma(shape=2.0, scale=5.0, size=(days, len(lats), len(lons)))
    precip_base = _smooth(precip_noise, sigma=3.0)

    # Modulate precip by regime
    regime_precip_mult = np.ones((days, 1, 1))
    regime_precip_mult[regime_labels == 0] = 3.0
    regime_precip_mult[regime_labels == 1] = 0.5
    precip_base = precip_base * regime_precip_mult

    # Moving blob (low pressure advection)
    blob = np.zeros((days, len(lats), len(lons)))
    for d in range(days):
        # blob moves linearly
        c_lat = 15.0 + (d % 30) * 0.5
        c_lon = 85.0 - (d % 30) * 0.5
        dist = np.sqrt((lat2d - c_lat)**2 + (lon2d - c_lon)**2)
        blob[d] = np.exp(-dist**2 / 10.0) * 20.0
    
    # Cyclone in BoB (regime 3 mostly)
    cyclone = np.zeros((days, len(lats), len(lons)))
    for d in range(days):
        if regime_labels[d] == 3 and d % 5 == 0:
            c_lat = 15.0 + np.random.rand()*5
            c_lon = 88.0 + np.random.rand()*3
            dist = np.sqrt((lat2d - c_lat)**2 + (lon2d - c_lon)**2)
            cyclone[d] = np.exp(-dist**2 / 5.0) * 50.0

    precip_truth = precip_base + (
        is_ghats * np.random.gamma(2.0, 15.0, size=(days, len(lats), len(lons)))
    ) + (is_ne * np.random.gamma(2.0, 10.0, size=(days, len(lats), len(lons)))) + blob + cyclone
    
    precip_truth = _smooth(precip_truth, 1.5)
    # Zero inflation
    zero_mask = precip_truth < 2.0
    precip_truth[zero_mask] = 0.0

    # Temp
    # N-S gradient
    t2m_grad = 35.0 - 0.5 * (lat2d - lat_min)
    t2m_noise = _smooth(np.random.normal(0, 3, size=(days, len(lats), len(lons))), 3.0)
    
    t2m_truth = t2m_grad + t2m_noise + (is_hot * 5.0) - (is_himalayas * 15.0)
    wd_cooling = np.zeros((days, 1, 1))
    wd_cooling[regime_labels == 2] = -5.0
    t2m_truth = t2m_truth + wd_cooling

    # Wind
    wind_truth = _smooth(np.random.weibull(2.0, size=(days, len(lats), len(lons))), 3.0) * 5.0
    wind_truth += (cyclone / 50.0) * 20.0  # high wind in cyclone
    gust_truth = wind_truth * 1.5 + _smooth(np.random.normal(1.0, 1.0, size=(days, len(lats), len(lons))), 2.0)
    gust_truth = np.maximum(gust_truth, wind_truth)

    truth_ds = xr.Dataset(
        {
            "precip": (["time", "lat", "lon"], precip_truth, {"units": "mm"}),
            "t2m": (["time", "lat", "lon"], t2m_truth, {"units": "C"}),
            "wind10m": (["time", "lat", "lon"], wind_truth, {"units": "m s-1"}),
            "gust10m": (["time", "lat", "lon"], gust_truth, {"units": "m s-1"}),
        },
        coords={
            "time": times,
            "lat": (["lat"], lats, {"units": "degrees_north"}),
            "lon": (["lon"], lons, {"units": "degrees_east"}),
        },
    )

    # 2. GENERATE MODELS
    models = {}
    det_model_names = ["ncum_g", "ecmwf_ifs", "gfs", "aifs", "graphcast", "pangu"]
    t_shape = (days, len(leads), len(lats), len(lons))

    precip_truth_expanded = np.expand_dims(truth_ds["precip"].values, axis=1).repeat(len(leads), axis=1)
    t2m_truth_expanded = np.expand_dims(truth_ds["t2m"].values, axis=1).repeat(len(leads), axis=1)
    wind_truth_expanded = np.expand_dims(truth_ds["wind10m"].values, axis=1).repeat(len(leads), axis=1)
    gust_truth_expanded = np.expand_dims(truth_ds["gust10m"].values, axis=1).repeat(len(leads), axis=1)

    for m in det_model_names:
        is_ai = m in ["aifs", "graphcast", "pangu"]
        lead_factor = np.arange(1, len(leads) + 1) / 5.0
        lead_error = lead_factor.reshape(1, len(leads), 1, 1)

        if is_ai:
            p_err = _smooth(np.random.normal(0, 3, size=t_shape), 3.0) * lead_error
            p_model = precip_truth_expanded * 0.8 + p_err
            t_err = _smooth(np.random.normal(0, 1.5, size=t_shape), 4.0) * lead_error
        else:
            p_err = _smooth(np.random.normal(0, 5, size=t_shape), 2.0) * lead_error
            p_model = precip_truth_expanded + p_err + np.expand_dims(is_ghats, axis=(0, 1)) * _smooth(np.random.normal(0, 2, size=t_shape), 2.0)
            t_err = _smooth(np.random.normal(0, 2.0, size=t_shape), 3.0) * (lead_error**1.5)

        p_model = np.maximum(p_model, 0.0)

        w_err = _smooth(np.random.normal(0, 1.5, size=t_shape), 3.0) * lead_error
        g_err = _smooth(np.random.normal(0, 2.5, size=t_shape), 3.0) * lead_error

        ds = xr.Dataset(
            {
                "precip": (["time", "lead", "lat", "lon"], p_model, {"units": "mm"}),
                "t2m": (["time", "lead", "lat", "lon"], t2m_truth_expanded + t_err, {"units": "C"}),
                "wind10m": (["time", "lead", "lat", "lon"], np.maximum(wind_truth_expanded + w_err, 0), {"units": "m s-1"}),
                "gust10m": (["time", "lead", "lat", "lon"], np.maximum(gust_truth_expanded + g_err, 0), {"units": "m s-1"}),
            },
            coords={
                "time": times,
                "lead": (["lead"], pd.to_timedelta(leads, unit="h")),
                "lat": (["lat"], lats, {"units": "degrees_north"}),
                "lon": (["lon"], lons, {"units": "degrees_east"}),
            },
        )
        models[m] = ds

    # 3. GENERATE ENSEMBLES
    ensembles = {}
    ens_model_names = ["ecmwf_ens", "neps"]
    n_members = 10

    for m in ens_model_names:
        lead_error = (np.arange(1, len(leads) + 1) / 4.0).reshape(1, len(leads), 1, 1, 1)
        e_shape = (days, len(leads), n_members, len(lats), len(lons))

        p_t = np.expand_dims(precip_truth_expanded, axis=2).repeat(n_members, axis=2)
        t_t = np.expand_dims(t2m_truth_expanded, axis=2).repeat(n_members, axis=2)
        w_t = np.expand_dims(wind_truth_expanded, axis=2).repeat(n_members, axis=2)
        g_t = np.expand_dims(gust_truth_expanded, axis=2).repeat(n_members, axis=2)

        # To keep generation fast, we smooth the member errors across lat/lon
        # But we must loop members or axes=(-2, -1) handles it if ndim>=2
        p_spread = _smooth(np.random.normal(0, 3, size=e_shape), 2.0) * lead_error
        t_spread = _smooth(np.random.normal(0, 1.5, size=e_shape), 3.0) * lead_error

        ds = xr.Dataset(
            {
                "precip": (["time", "lead", "member", "lat", "lon"], np.maximum(p_t + p_spread, 0), {"units": "mm"}),
                "t2m": (["time", "lead", "member", "lat", "lon"], t_t + t_spread, {"units": "C"}),
                "wind10m": (["time", "lead", "member", "lat", "lon"], np.maximum(w_t + _smooth(np.random.normal(0, 1, e_shape), 2.0), 0), {"units": "m s-1"}),
                "gust10m": (["time", "lead", "member", "lat", "lon"], np.maximum(g_t + _smooth(np.random.normal(0, 1.5, e_shape), 2.0), 0), {"units": "m s-1"}),
            },
            coords={
                "time": times,
                "lead": (["lead"], pd.to_timedelta(leads, unit="h")),
                "member": (["member"], np.arange(n_members)),
                "lat": (["lat"], lats, {"units": "degrees_north"}),
                "lon": (["lon"], lons, {"units": "degrees_east"}),
            },
        )
        ensembles[m] = ds

    return truth_ds, models, ensembles, regimes_ds

def run_synthetic_pipeline(out_dir="data/demo", days=90):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    os.makedirs(f"{out_dir}/ensembles", exist_ok=True)

    truth, models, ensembles, regimes = generate_synthetic_data(days=days)

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Writing truth...")
    truth.to_zarr(f"{out_dir}/truth.zarr", mode="w")
    regimes.to_zarr(f"{out_dir}/regimes.zarr", mode="w")

    logger.info("Writing models...")
    for m, ds in models.items():
        ds.to_zarr(f"{out_dir}/models/{m}.zarr", mode="w")

    logger.info("Writing ensembles...")
    for m, ds in ensembles.items():
        ds.to_zarr(f"{out_dir}/ensembles/{m}.zarr", mode="w")

    logger.info("Synthetic data generation complete.")

if __name__ == "__main__":
    run_synthetic_pipeline()
