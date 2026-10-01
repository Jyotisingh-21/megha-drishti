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
