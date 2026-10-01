import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nwpblend.extremes.cyclone import cyclone_consensus
from nwpblend.extremes.rain2stage import TwoStagePrecipModel
from nwpblend.extremes.thresholds import get_heatwave_thresholds, get_precip_thresholds


def test_thresholds_loaded():
    pt = get_precip_thresholds()
    assert pt["heavy"] == 64.5
    assert pt["very_heavy"] == 115.6
    assert pt["extremely_heavy"] == 204.5
    # Check ordering
    assert pt["heavy"] < pt["very_heavy"] < pt["extremely_heavy"]

    ht = get_heatwave_thresholds()
    assert ht["absolute"]["plains"] == 40.0
    assert ht["absolute"]["hilly"] == 30.0


def test_rain2stage_non_negative_and_monotone():
    # Setup tiny mock data
    N = 100
    mean_val = np.random.gamma(2, 5, size=N)
    truth_val = mean_val + np.random.normal(0, 2, size=N)
    truth_val = np.clip(truth_val, 0, None)

    X_ds = xr.Dataset({"precip": (["time"], mean_val)})
    truth_da = xr.DataArray(truth_val, dims=["time"])

    model = TwoStagePrecipModel()
    model.fit(X_ds, truth_da)

    ds = model.predict_distribution(X_ds)
    # Output must be non-negative
    assert np.all(ds["conditional_amount"].values >= 0)
    assert np.all(ds["expected_amount"].values >= 0)
    assert np.all((ds["pop"].values >= 0) & (ds["pop"].values <= 1))

    # Exceedance should be monotonically decreasing
    p_10 = model.exceedance_prob(10.0, X_ds).values
    p_50 = model.exceedance_prob(50.0, X_ds).values
    assert np.all(p_10 >= p_50)


def test_cyclone_consensus_zero_spread():
    df = pd.DataFrame(
        {
            "model": ["m1", "m2", "m3"],
            "time": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "lat": [15.0, 15.0, 15.0],
            "lon": [70.0, 70.0, 70.0],
        }
    )

    res = cyclone_consensus(df)
    assert len(res) == 1
    assert res["mean_lat"].iloc[0] == 15.0
    assert res["mean_lon"].iloc[0] == 70.0
    assert res["spread_km"].iloc[0] == 0.0
