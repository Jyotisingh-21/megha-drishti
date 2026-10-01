import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.biascorrect.core import BiasCorrector
from nwpblend.biascorrect.lapse_rate import correct_t2m
from nwpblend.biascorrect.quantile_map import QuantileMapper


def test_lapse_rate():
    ds = xr.Dataset({"t2m": (["lat", "lon"], np.array([[300.0]]))})
    oro_model = xr.DataArray(np.array([[1000.0]]))
    oro_truth = xr.DataArray(np.array([[0.0]]))

    # Model is higher, so it's artificially colder. We add temp.
    # lapse_rate = 0.0065
    # diff = 1000 - 0 = 1000
    # correction = 1000 * 0.0065 = 6.5
    # Expected: 306.5

    out = correct_t2m(ds, oro_model, oro_truth)
    assert np.allclose(out["t2m"].values, 306.5)


def test_quantile_mapper_monotone():
    # Synthetic data
    truth = xr.Dataset(
        {"t2m": (["time", "lat", "lon"], np.random.normal(30, 2, (100, 1, 1)))},
        coords={"time": pd.date_range("2024-01-01", periods=100), "lat": [10], "lon": [70]},
    )

    fcst = xr.Dataset(
        {"t2m": (["time", "lead", "lat", "lon"], np.random.normal(25, 5, (100, 1, 1, 1)))},
        coords={
            "time": pd.date_range("2024-01-01", periods=100),
            "lead": [24],
            "lat": [10],
            "lon": [70],
        },
    )

    qm = QuantileMapper()
    qm.fit(fcst, truth)

    test_fcst = xr.Dataset(
        {
            "t2m": (
                ["time", "lead", "lat", "lon"],
                np.array(
                    [
                        [[[10.0]]],
                        [[[20.0]]],
                        [[[30.0]]],
                        [[[40.0]]],
                        [[[50.0]]],
                        [[[60.0]]],
                        [[[70.0]]],
                    ]
                ),
            )
        },
        coords={
            "time": pd.date_range("2024-05-01", periods=7),
            "lead": [24],
            "lat": [10],
            "lon": [70],
        },
    )

    out = qm.transform(test_fcst)
    vals = out["t2m"].values.flatten()

    # Monotonicity check
    assert np.all(np.diff(vals) >= 0)


def test_quantile_mapper_precip_non_negative():
    truth = xr.Dataset(
        {"precip": (["time", "lat", "lon"], np.random.exponential(5, (100, 1, 1)))},
        coords={"time": pd.date_range("2024-01-01", periods=100), "lat": [10], "lon": [70]},
    )

    fcst = xr.Dataset(
        {
            "precip": (["time", "lead", "lat", "lon"], np.random.normal(5, 5, (100, 1, 1, 1)))
        },  # Can be negative in normal
        coords={
            "time": pd.date_range("2024-01-01", periods=100),
            "lead": [24],
            "lat": [10],
            "lon": [70],
        },
    )

    qm = QuantileMapper(precip_threshold=0.1)
    qm.fit(fcst, truth)

    test_fcst = xr.Dataset(
        {
            "precip": (
                ["time", "lead", "lat", "lon"],
                np.array([[[[-10.0]]], [[[-1.0]]], [[[0.0]]], [[[5.0]]], [[[20.0]]]]),
            )
        },
        coords={
            "time": pd.date_range("2024-05-01", periods=5),
            "lead": [24],
            "lat": [10],
            "lon": [70],
        },
    )

    out = qm.transform(test_fcst)
    vals = out["precip"].values.flatten()

    assert np.all(vals >= 0.0)
