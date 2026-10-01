import numpy as np
import xarray as xr

from nwpblend.verify.metrics import bias, brier_score, contingency_scores, mae, rmse


def test_metrics():
    # Construct small test arrays
    fcst = xr.DataArray([1.0, 2.0, 3.0], dims=["time"])
    truth = xr.DataArray([1.0, 1.0, 4.0], dims=["time"])

    b = bias(fcst, truth)
    assert np.isclose(b.item(), (0.0 + 1.0 - 1.0) / 3.0)

    m = mae(fcst, truth)
    assert np.isclose(m.item(), (0.0 + 1.0 + 1.0) / 3.0)

    r = rmse(fcst, truth)
    assert np.isclose(r.item(), np.sqrt((0.0 + 1.0 + 1.0) / 3.0))


def test_contingency_scores():
    fb = xr.DataArray([True, True, False, False], dims=["time"])
    tb = xr.DataArray([True, False, True, False], dims=["time"])

    pod, far, csi = contingency_scores(fb, tb)

    # hits: 1 (idx 0)
    # misses: 1 (idx 2)
    # false_alarms: 1 (idx 1)

    assert np.isclose(pod.item(), 1 / (1 + 1))
    assert np.isclose(far.item(), 1 / (1 + 1))
    assert np.isclose(csi.item(), 1 / (1 + 1 + 1))
