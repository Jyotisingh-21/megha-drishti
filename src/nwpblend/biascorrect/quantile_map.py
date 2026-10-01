import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class QuantileMapper:
    def __init__(self, n_quantiles=100, precip_threshold=0.1):
        self.n_quantiles = n_quantiles
        self.precip_threshold = precip_threshold
        self.quantiles = np.linspace(0, 1, self.n_quantiles)
        self.fit_params = {}

    def _fit_1d(self, fcst: np.ndarray, truth: np.ndarray, is_precip: bool):
        # Remove NaNs
        fcst = fcst[~np.isnan(fcst)]
        truth = truth[~np.isnan(truth)]

        if len(fcst) == 0 or len(truth) == 0:
            return None

        if is_precip:
            truth_wet = truth[truth >= self.precip_threshold]
            p_wet_truth = len(truth_wet) / len(truth) if len(truth) > 0 else 0.0
            
            if p_wet_truth > 0:
                fcst_thresh = np.quantile(fcst, max(0.0, 1.0 - p_wet_truth))
                fcst_wet_adj = fcst[fcst >= fcst_thresh]
                
                if len(fcst_wet_adj) > 0:
                    q_fcst = np.quantile(fcst_wet_adj, self.quantiles)
                    q_truth = np.quantile(truth_wet, self.quantiles)
                else:
                    q_fcst = np.zeros_like(self.quantiles)
                    q_truth = np.zeros_like(self.quantiles)
            else:
                fcst_thresh = np.inf
                q_fcst = np.zeros_like(self.quantiles)
                q_truth = np.zeros_like(self.quantiles)
                
            return {
                "fcst_thresh": fcst_thresh,
                "q_fcst": q_fcst,
                "q_truth": q_truth
            }
        else:
            q_fcst = np.quantile(fcst, self.quantiles)
            q_truth = np.quantile(truth, self.quantiles)
            return {"q_fcst": q_fcst, "q_truth": q_truth}

    def fit(self, ds_fcst: xr.Dataset, ds_truth: xr.Dataset):
        """
        Fit EQM parameters per variable, lead, lat, lon.
        Currently operates in memory using numpy for simplicity on demo scale.
        """
        for var in ds_fcst.data_vars:
            is_precip = "precip" in var.lower()
            self.fit_params[var] = {}

            # For each lead, lat, lon
            if "lead" in ds_fcst.coords:
                for lead in ds_fcst.lead.values:
                    self.fit_params[var][lead] = {}

                    fcst_sel = ds_fcst[var].sel(lead=lead).values
                    truth_sel = ds_truth[var].values  # Truth typically doesn't have lead

                    # Assuming dimensions are (time, lat, lon)
                    # We will loop over lat and lon. For massive arrays, this should be vectorized via xr.apply_ufunc.
                    # For prototype, a simple loop or vectorization.

                    lats = ds_fcst.lat.values
                    lons = ds_fcst.lon.values

                    # Store as arrays for fast access
                    params = np.empty((len(lats), len(lons)), dtype=object)

                    for i in range(len(lats)):
                        for j in range(len(lons)):
                            f_ts = fcst_sel[:, i, j]
                            t_ts = truth_sel[:, i, j]
                            params[i, j] = self._fit_1d(f_ts, t_ts, is_precip)

                    self.fit_params[var][lead] = params
            else:
                # No lead dimension
                pass

    def _transform_1d(self, fcst: np.ndarray, params: dict, is_precip: bool):
        if params is None:
            return fcst

        if is_precip:
            thresh = params.get("fcst_thresh", np.inf)
            out = np.zeros_like(fcst)
            wet_mask = fcst >= thresh
            if np.any(wet_mask):
                out[wet_mask] = np.interp(fcst[wet_mask], params["q_fcst"], params["q_truth"])
            return out
        else:
            q_fcst = params["q_fcst"]
            q_truth = params["q_truth"]
            # Constant extrapolation at bounds
            return np.interp(fcst, q_fcst, q_truth)

    def transform(self, ds_fcst: xr.Dataset) -> xr.Dataset:
        out = ds_fcst.load().copy(deep=True)

        for var in out.data_vars:
            if var not in self.fit_params:
                continue

            is_precip = "precip" in var.lower()

            if "lead" in out.coords:
                for lead in out.lead.values:
                    params_grid = self.fit_params[var].get(lead)
                    if params_grid is None:
                        continue

                    arr = out[var].sel(lead=lead).values
                    lats = out.lat.values
                    lons = out.lon.values

                    for i in range(len(lats)):
                        for j in range(len(lons)):
                            p = params_grid[i, j]
                            arr[:, i, j] = self._transform_1d(arr[:, i, j], p, is_precip)

                    out[var].loc[{"lead": lead}] = arr
        return out
