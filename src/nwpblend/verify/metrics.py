import numpy as np
import scipy.signal
import xarray as xr
from scipy.stats import norm


def bias(fcst: xr.DataArray, truth: xr.DataArray, dim="time") -> xr.DataArray:
    return (fcst - truth).mean(dim=dim)


def mae(fcst: xr.DataArray, truth: xr.DataArray, dim="time") -> xr.DataArray:
    return np.abs(fcst - truth).mean(dim=dim)


def rmse(fcst: xr.DataArray, truth: xr.DataArray, dim="time") -> xr.DataArray:
    return np.sqrt(((fcst - truth) ** 2).mean(dim=dim))


def acc(fcst: xr.DataArray, truth: xr.DataArray, climo: xr.DataArray, dim="time") -> xr.DataArray:
    f_anom = fcst - climo
    t_anom = truth - climo

    cov = (f_anom * t_anom).mean(dim=dim)
    std_f = np.sqrt((f_anom**2).mean(dim=dim))
    std_t = np.sqrt((t_anom**2).mean(dim=dim))

    return cov / (std_f * std_t)


def brier_score(prob_fcst: xr.DataArray, truth_binary: xr.DataArray, dim="time") -> xr.DataArray:
    return ((prob_fcst - truth_binary) ** 2).mean(dim=dim)


def crps_gaussian(mu: xr.DataArray, sigma: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    """CRPS for a Gaussian distribution with analytical formula."""
    # z = (truth - mu) / sigma
    # crps = sigma * (z * (2 * CDF(z) - 1) + 2 * PDF(z) - 1 / sqrt(pi))
    # Handle sigma = 0 safely
    sigma = xr.where(sigma == 0, 1e-8, sigma)
    z = (truth - mu) / sigma

    pdf_z = xr.apply_ufunc(norm.pdf, z)
    cdf_z = xr.apply_ufunc(norm.cdf, z)

    crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
    return crps.mean(dim="time")


def contingency_scores(fcst_binary: xr.DataArray, truth_binary: xr.DataArray, dim="time"):
    hits = (fcst_binary & truth_binary).sum(dim=dim)
    misses = (~fcst_binary & truth_binary).sum(dim=dim)
    false_alarms = (fcst_binary & ~truth_binary).sum(dim=dim)

    pod = hits / (hits + misses)
    far = false_alarms / (hits + false_alarms)
    csi = hits / (hits + misses + false_alarms)

    return pod, far, csi


def _convolve_2d(arr, window):
    kernel = np.ones((window, window))
    return scipy.signal.convolve2d(arr, kernel, mode="same", boundary="fill", fillvalue=0)


def fss(fcst: xr.DataArray, truth: xr.DataArray, threshold: float, window: int = 3) -> xr.DataArray:
    """
    Fractions Skill Score over lat/lon grids.
    Requires 'lat' and 'lon' dimensions.
    Returns FSS scalar per time/lead step, then averaged over time.
    """
    fb = (fcst >= threshold).astype(float)
    tb = (truth >= threshold).astype(float)

    def calc_fss_2d(f, t):
        if np.isnan(f).all() or np.isnan(t).all():
            return np.nan
        # Fill nans for convolution
        f = np.nan_to_num(f)
        t = np.nan_to_num(t)
        p_f = _convolve_2d(f, window) / (window * window)
        p_t = _convolve_2d(t, window) / (window * window)

        mse = np.nanmean((p_f - p_t) ** 2)
        ref_mse = np.nanmean(p_f**2) + np.nanmean(p_t**2)

        if ref_mse == 0:
            return 1.0 if mse == 0 else 0.0
        return 1.0 - (mse / ref_mse)

    # Apply over lat/lon for each time/lead slice
    # For a full xarray implementation without loops, we'd use xr.apply_ufunc with dask.
    # Here we use a simpler loop or apply_ufunc signature.
    res = xr.apply_ufunc(
        calc_fss_2d,
        fb,
        tb,
        input_core_dims=[["lat", "lon"], ["lat", "lon"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    return res.mean(dim="time")
