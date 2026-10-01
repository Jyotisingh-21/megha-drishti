import numpy as np
import xarray as xr

from nwpblend.blend.emos import EMOSCalibrator


def pit_histogram(truth: np.ndarray, mean: np.ndarray, std: np.ndarray, bins=10):
    from scipy.stats import norm

    # PIT = CDF(truth | mean, std)
    z = (truth - mean) / np.clip(std, 1e-6, None)
    pits = norm.cdf(z)
    hist, _ = np.histogram(pits, bins=bins, range=(0, 1))
    return hist


def test_emos_gaussian():
    # Synthetic truth
    N = 1000
    truth_val = np.random.normal(20, 5, size=N)

    # Deliberately under-dispersive ensemble
    # Mean is truth + error, spread is tiny
    mean_val = truth_val + np.random.normal(0, 2, size=N)
    spread_val = np.ones(N) * 0.5  # true error is ~2.0

    mean_da = xr.DataArray(mean_val, dims=["time"])
    spread_da = xr.DataArray(spread_val, dims=["time"])

    calibrator = EMOSCalibrator(is_precip=False)
    calibrator.fit(mean_val, spread_val, truth_val)

    ds = calibrator.predict(mean_da, spread_da)
    cal_spread = ds["calibrated_spread"].values

    # Check that spread increased since it was under-dispersive
    assert np.mean(cal_spread) > np.mean(spread_val) * 2.0

    # Check exceedance
    prob = calibrator.exceedance_prob(20.0, mean_da, spread_da)
    assert np.all((prob >= 0) & (prob <= 1))


def test_emos_precip():
    N = 1000
    # Zero inflated truth
    rain = np.random.rand(N) > 0.7
    truth_val = np.zeros(N)
    truth_val[rain] = np.random.gamma(2, 5, size=np.sum(rain))

    # Model predicts poorly
    mean_val = truth_val * 0.5 + np.random.normal(1, 1, size=N)
    mean_val = np.clip(mean_val, 0, None)
    spread_val = np.ones(N)

    mean_da = xr.DataArray(mean_val, dims=["time"])
    spread_da = xr.DataArray(spread_val, dims=["time"])

    calibrator = EMOSCalibrator(is_precip=True)
    calibrator.fit(mean_val, spread_val, truth_val)

    ds = calibrator.predict(mean_da, spread_da)

    prob = calibrator.exceedance_prob(5.0, mean_da, spread_da)
    assert np.all((prob >= 0) & (prob <= 1))

    # 90th percentile should be >= 50th percentile
    assert np.all(ds["q90"].values >= ds["q50"].values)
