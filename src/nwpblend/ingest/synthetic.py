import numpy as np
import pandas as pd
import xarray as xr


def generate_synthetic_data(
    lat_min=6.0, lat_max=37.0, lon_min=68.0, lon_max=98.0, resolution=1.0, days=90
):
    """
    Generate synthetic truth, deterministic models, and ensemble models for the demo.
    """
    np.random.seed(42)

    lats = np.arange(lat_min, lat_max + resolution / 2, resolution)
    lons = np.arange(lon_min, lon_max + resolution / 2, resolution)

    # 90 days of initializations
    times = pd.date_range("2024-06-01", periods=days, freq="D")

    # Leads 1 to 10 days (in hours)
    leads = np.arange(24, 24 * 10 + 24, 24)

    # 1. Regimes
    # 0: monsoon-active, 1: monsoon-break, 2: western-disturbance, 3: other
    regime_labels = np.random.choice([0, 1, 2, 3], size=days, p=[0.4, 0.3, 0.1, 0.2])
    regimes_da = xr.DataArray(regime_labels, dims=["time"], coords={"time": times}, name="regime")
    regimes_ds = xr.Dataset({"regime": regimes_da})

    # Spatial mesh for patterns
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Western Ghats proxy: roughly lon 73-77, lat 10-20
    is_ghats = ((lon2d >= 73) & (lon2d <= 77) & (lat2d >= 10) & (lat2d <= 20)).astype(float)

    # 2. Generate Truth
    # Precip: gamma distributed, mostly zeros
    precip_base = np.random.gamma(shape=2.0, scale=10.0, size=(days, len(lats), len(lons)))
    zero_mask = np.random.rand(days, len(lats), len(lons)) > 0.3
    precip_base[zero_mask] = 0.0

    # Enhance precip over Ghats
    precip_truth = precip_base + (
        is_ghats * np.random.gamma(2.0, 20.0, size=(days, len(lats), len(lons)))
    )

    # Temp
    t2m_truth = (
        25.0
        + 10.0 * np.cos(np.deg2rad(lat2d - 20))
        + np.random.normal(0, 2, size=(days, len(lats), len(lons)))
    )

    # Winds
    wind_truth = np.random.weibull(2.0, size=(days, len(lats), len(lons))) * 5.0
    gust_truth = wind_truth * 1.5 + np.random.normal(1.0, 0.5, size=(days, len(lats), len(lons)))
    gust_truth = np.maximum(gust_truth, wind_truth)  # Gust must be >= wind

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

    # 3. Generate Models
    models = {}
    det_model_names = ["ncum_g", "ecmwf_ifs", "gfs", "aifs", "graphcast", "pangu"]

    # Pre-allocate base truth fields across leads for easy addition
    t_shape = (days, len(leads), len(lats), len(lons))

    precip_truth_expanded = np.expand_dims(truth_ds["precip"].values, axis=1).repeat(
        len(leads), axis=1
    )
    t2m_truth_expanded = np.expand_dims(truth_ds["t2m"].values, axis=1).repeat(len(leads), axis=1)
    wind_truth_expanded = np.expand_dims(truth_ds["wind10m"].values, axis=1).repeat(
        len(leads), axis=1
    )
    gust_truth_expanded = np.expand_dims(truth_ds["gust10m"].values, axis=1).repeat(
        len(leads), axis=1
    )

    for m in det_model_names:
        is_ai = m in ["aifs", "graphcast", "pangu"]

        # Error grows with lead:
        # AI models have lower lead-growth for temperature, but under-predict precip extremes.
        # NWP models have higher variance at long leads but better extremes.
        lead_factor = np.arange(1, len(leads) + 1) / 5.0
        lead_error = lead_factor.reshape(1, len(leads), 1, 1)

        # Specific model weaknesses
        if is_ai:
            p_err = np.random.normal(0, 3, size=t_shape) * lead_error
            # AI models tend to smooth, so we dampen the truth slightly
            p_model = precip_truth_expanded * 0.8 + p_err
            t_err = np.random.normal(0, 1.0, size=t_shape) * lead_error
        else:
            p_err = np.random.normal(0, 5, size=t_shape) * lead_error
            # NWP models are good at extremes, especially over Ghats
            p_model = (
                precip_truth_expanded
                + p_err
                + np.expand_dims(is_ghats, axis=(0, 1)) * np.random.normal(0, 2, size=t_shape)
            )
            t_err = np.random.normal(0, 1.5, size=t_shape) * (lead_error**1.5)

        p_model = np.maximum(p_model, 0.0)  # No negative rain

        w_err = np.random.normal(0, 1.0, size=t_shape) * lead_error
        g_err = np.random.normal(0, 2.0, size=t_shape) * lead_error

        ds = xr.Dataset(
            {
                "precip": (["time", "lead", "lat", "lon"], p_model, {"units": "mm"}),
                "t2m": (["time", "lead", "lat", "lon"], t2m_truth_expanded + t_err, {"units": "C"}),
                "wind10m": (
                    ["time", "lead", "lat", "lon"],
                    np.maximum(wind_truth_expanded + w_err, 0),
                    {"units": "m s-1"},
                ),
                "gust10m": (
                    ["time", "lead", "lat", "lon"],
                    np.maximum(gust_truth_expanded + g_err, 0),
                    {"units": "m s-1"},
                ),
            },
            coords={
                "time": times,
                "lead": (["lead"], pd.to_timedelta(leads, unit="h")),
                "lat": (["lat"], lats, {"units": "degrees_north"}),
                "lon": (["lon"], lons, {"units": "degrees_east"}),
            },
        )
        models[m] = ds

    # 4. Generate Ensembles
    ensembles = {}
    ens_model_names = ["ecmwf_ens", "neps"]
    n_members = 10

    for m in ens_model_names:
        # Base error
        lead_error = (np.arange(1, len(leads) + 1) / 4.0).reshape(1, len(leads), 1, 1, 1)
        e_shape = (days, len(leads), n_members, len(lats), len(lons))

        # Expand truth
        p_t = np.expand_dims(precip_truth_expanded, axis=2).repeat(n_members, axis=2)
        t_t = np.expand_dims(t2m_truth_expanded, axis=2).repeat(n_members, axis=2)
        w_t = np.expand_dims(wind_truth_expanded, axis=2).repeat(n_members, axis=2)
        g_t = np.expand_dims(gust_truth_expanded, axis=2).repeat(n_members, axis=2)

        # Spread grows with lead
        p_spread = np.random.normal(0, 3, size=e_shape) * lead_error
        t_spread = np.random.normal(0, 1.5, size=e_shape) * lead_error

        ds = xr.Dataset(
            {
                "precip": (
                    ["time", "lead", "member", "lat", "lon"],
                    np.maximum(p_t + p_spread, 0),
                    {"units": "mm"},
                ),
                "t2m": (["time", "lead", "member", "lat", "lon"], t_t + t_spread, {"units": "C"}),
                "wind10m": (
                    ["time", "lead", "member", "lat", "lon"],
                    np.maximum(w_t + np.random.normal(0, 1, e_shape), 0),
                    {"units": "m s-1"},
                ),
                "gust10m": (
                    ["time", "lead", "member", "lat", "lon"],
                    np.maximum(g_t + np.random.normal(0, 1.5, e_shape), 0),
                    {"units": "m s-1"},
                ),
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
