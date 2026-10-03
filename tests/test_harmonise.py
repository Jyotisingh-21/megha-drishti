import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nwpblend.harmonise.regrid import HAS_XESMF, regrid, wrap_longitudes
from nwpblend.harmonise.store import stack_models


def test_wrap_longitudes():
    ds = xr.Dataset(coords={"lon": np.array([350, 355, 0, 5])})
    wrapped = wrap_longitudes(ds)
    assert np.allclose(wrapped.lon.values, [-10, -5, 0, 5])


def test_regrid_conservation():
    # Only strictly check if xesmf is available
    if not HAS_XESMF:
        pytest.skip("xESMF not available, skipping conservative regrid test")

    ds = xr.Dataset(
        {"precip": (["lat", "lon"], np.ones((10, 10)) * 5.0)},
        coords={
            "lat": np.linspace(10, 20, 10),
            "lon": np.linspace(70, 80, 10),
        },
    )

    target_grid = xr.Dataset(
        coords={
            "lat": np.linspace(10, 20, 5),
            "lon": np.linspace(70, 80, 5),
        }
    )

    out = regrid(ds, target_grid)

    mean_in = ds["precip"].mean().item()
    mean_out = out["precip"].mean().item()

    # Area-mean precip conservation within a tolerance
    np.testing.assert_allclose(mean_in, mean_out, rtol=1e-2)


def test_stack_models():
    ds1 = xr.Dataset(
        {"t2m": (["time", "lat", "lon"], np.ones((1, 2, 2)))},
        coords={"time": [pd.to_datetime("2024-07-30")], "lat": [10, 20], "lon": [70, 80]},
    )

    ds2 = ds1 * 2

    models = {"ecmwf_ifs": ds1, "gfs": ds2}
    expected = ["ecmwf_ifs", "gfs", "aifs"]

    stacked = stack_models(models, expected)

    assert "model" in stacked.dims
    assert len(stacked.model) == 3

    # Check values
    assert np.allclose(stacked.sel(model="ecmwf_ifs")["t2m"].values, 1.0)
    assert np.allclose(stacked.sel(model="gfs")["t2m"].values, 2.0)

    # Missing model should be NaN
    assert np.isnan(stacked.sel(model="aifs")["t2m"].values).all()

    # Check available mask
    assert "available" in stacked
    av = stacked["available"].sel(time="2024-07-30")
    assert av.sel(model="ecmwf_ifs").item() is True
    assert av.sel(model="gfs").item() is True
    assert av.sel(model="aifs").item() is False


def test_stack_models_varying_leads():
    time = pd.date_range("2024-01-01", periods=1)
    lat = np.array([10.0])
    lon = np.array([77.0])

    # Model with leads [24, 48, 72]
    dsA = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.ones((1, 3, 1, 1)))},
        coords={"time": time, "lead": [24, 48, 72], "lat": lat, "lon": lon},
    )

    # Model with leads [24]
    dsB = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.ones((1, 1, 1, 1)) * 2)},
        coords={"time": time, "lead": [24], "lat": lat, "lon": lon},
    )

    # Model with leads [72]
    dsC = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.ones((1, 1, 1, 1)) * 3)},
        coords={"time": time, "lead": [72], "lat": lat, "lon": lon},
    )

    # Model with NO leads (zero length dim)
    dsD = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.zeros((1, 0, 1, 1)))},
        coords={"time": time, "lead": [], "lat": lat, "lon": lon},
    )

    # 1. Multiple models with varying leads
    models = {"ecmwf": dsA, "gfs": dsB, "ncum": dsC, "aifs": dsD}
    expected = ["ecmwf", "gfs", "ncum", "aifs", "missing_model"]

    stacked = stack_models(models, expected)

    assert list(stacked.lead.values) == [24, 48, 72]
    assert stacked.dims["model"] == 5

    # Check mask per model, lead
    av = stacked["available"].sel(time="2024-01-01")

    assert av.sel(model="ecmwf", lead=24).item() is True
    assert av.sel(model="ecmwf", lead=48).item() is True
    assert av.sel(model="ecmwf", lead=72).item() is True

    assert av.sel(model="gfs", lead=24).item() is True
    assert av.sel(model="gfs", lead=48).item() is False
    assert av.sel(model="gfs", lead=72).item() is False

    assert av.sel(model="ncum", lead=24).item() is False
    assert av.sel(model="ncum", lead=48).item() is False
    assert av.sel(model="ncum", lead=72).item() is True

    assert av.sel(model="aifs", lead=24).item() is False
    assert av.sel(model="aifs", lead=72).item() is False

    assert av.sel(model="missing_model", lead=24).item() is False

    # 2. Case with a single model only
    models_single = {"ecmwf": dsA}
    stacked_single = stack_models(models_single, expected)

    assert list(stacked_single.lead.values) == [24, 48, 72]
    av_single = stacked_single["available"].sel(time="2024-01-01")
    assert av_single.sel(model="ecmwf", lead=24).item() is True
    assert av_single.sel(model="gfs", lead=24).item() is False
