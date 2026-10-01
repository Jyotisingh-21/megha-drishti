# Removed legacy typing

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import norm


class EMOSCalibrator:
    """
    Non-homogeneous Gaussian regression (EMOS) for probabilistic calibration.
    """

    def __init__(self, is_precip: bool = False, precip_threshold: float = 0.1):
        self.is_precip = is_precip
        self.precip_threshold = precip_threshold
        # Params: [a, b, c, d]
        # mu = a + b * mean
        # var = c + d * var_ens
        self.params = np.array([0.0, 1.0, 0.1, 1.0])
        # For precip zero-mass logistic: [w0, w1] for logit(P(rain))
        self.precip_logistic = np.array([0.0, 1.0])
        self.is_fitted = False

    def _crps_gaussian_loss(self, params, mean_fcst, var_fcst, truth):
        a, b, c, d = params
        mu = a + b * mean_fcst
        # var must be positive
        sigma = np.sqrt(np.clip(c + d * var_fcst, 1e-6, np.inf))

        # Standardized error
        z = (truth - mu) / sigma

        pdf = norm.pdf(z)
        cdf = norm.cdf(z)

        crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        return np.mean(crps)

    def _logistic_loss(self, params, mean_fcst, rain_occurred):
        w0, w1 = params
        logits = w0 + w1 * mean_fcst
        # Binary Cross Entropy
        # probs = 1 / (1 + exp(-logits))
        # BCE = -y*log(p) - (1-y)*log(1-p)
        # Using softplus for stability: log(1+exp(logits)) - y*logits
        bce = np.log1p(np.exp(np.clip(logits, -20, 20))) - rain_occurred * logits
        return np.mean(bce)

    def fit(self, mean_fcst: np.ndarray, spread_fcst: np.ndarray, truth: np.ndarray):
        mean_fcst = mean_fcst.flatten()
        spread_fcst = spread_fcst.flatten()
        truth = truth.flatten()

        # Remove NaNs
        mask = ~np.isnan(mean_fcst) & ~np.isnan(spread_fcst) & ~np.isnan(truth)
        mean_fcst = mean_fcst[mask]
        var_fcst = spread_fcst[mask] ** 2
        truth = truth[mask]

        if len(truth) < 10:
            return self

        if not self.is_precip:
            res = minimize(
                self._crps_gaussian_loss,
                x0=self.params,
                args=(mean_fcst, var_fcst, truth),
                bounds=[(None, None), (None, None), (1e-6, None), (1e-6, None)],
                method="L-BFGS-B",
            )
            self.params = res.x
        else:
            # 1. Logistic for occurrence
            rain_occurred = (truth >= self.precip_threshold).astype(float)
            res_log = minimize(
                self._logistic_loss,
                x0=self.precip_logistic,
                args=(mean_fcst, rain_occurred),
                method="L-BFGS-B",
            )
            self.precip_logistic = res_log.x

            # 2. Gaussian on sqrt for amount (conditional on rain)
            wet_mask = rain_occurred > 0
            if np.sum(wet_mask) > 10:
                res_amt = minimize(
                    self._crps_gaussian_loss,
                    x0=self.params,
                    args=(
                        np.sqrt(np.clip(mean_fcst[wet_mask], 0, None)),
                        var_fcst[wet_mask],
                        np.sqrt(truth[wet_mask]),
                    ),
                    bounds=[(None, None), (None, None), (1e-6, None), (1e-6, None)],
                    method="L-BFGS-B",
                )
                self.params = res_amt.x

        self.is_fitted = True
        return self

    def _get_mu_sigma(
        self, mean_fcst: np.ndarray, spread_fcst: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b, c, d = self.params
        if self.is_precip:
            mu = a + b * np.sqrt(np.clip(mean_fcst, 0, None))
        else:
            mu = a + b * mean_fcst
        sigma = np.sqrt(np.clip(c + d * (spread_fcst**2), 1e-6, np.inf))
        return mu, sigma

    def predict(self, mean_da: xr.DataArray, spread_da: xr.DataArray) -> xr.Dataset:
        """
        Returns calibrated mean, spread, and quantiles (10, 50, 90).
        """
        mu, sigma = self._get_mu_sigma(mean_da.values, spread_da.values)

        if not self.is_precip:
            mean_val = mu
            spread_val = sigma
            q10 = norm.ppf(0.1, loc=mu, scale=sigma)
            q50 = mu
            q90 = norm.ppf(0.9, loc=mu, scale=sigma)
        else:
            w0, w1 = self.precip_logistic
            logits = w0 + w1 * mean_da.values
            prob_rain = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

            # Expected value of conditional sqrt-Gaussian
            # E[Y] approx (mu)^2 + sigma^2
            mean_wet = mu**2 + sigma**2
            mean_val = prob_rain * mean_wet
            # Spread is complex, proxy with conditional spread
            spread_val = sigma * 2.0 * np.abs(mu)  # delta method approx

            # Quantiles using mixture
            # F(x) = (1 - p) + p * Phi((sqrt(x) - mu)/sigma)
            # F(x) = q <=> Phi(...) = (q - (1-p))/p
            def _quantile(q):
                q_adj = (q - (1.0 - prob_rain)) / np.clip(prob_rain, 1e-6, 1.0)
                # If q_adj <= 0, quantile is 0
                q_adj = np.clip(q_adj, 1e-6, 1.0 - 1e-6)
                z = norm.ppf(q_adj)
                val = np.maximum(0.0, mu + sigma * z) ** 2
                return np.where(q <= (1.0 - prob_rain), 0.0, val)

            q10 = _quantile(0.10)
            q50 = _quantile(0.50)
            q90 = _quantile(0.90)

        ds = xr.Dataset(
            {
                "calibrated_mean": (mean_da.dims, mean_val),
                "calibrated_spread": (mean_da.dims, spread_val),
                "q10": (mean_da.dims, q10),
                "q50": (mean_da.dims, q50),
                "q90": (mean_da.dims, q90),
            },
            coords=mean_da.coords,
        )

        # Save state for exceedance_prob
        self._last_mean = mean_da
        self._last_spread = spread_da

        return ds

    def exceedance_prob(
        self, threshold: float, mean_da: xr.DataArray, spread_da: xr.DataArray
    ) -> xr.DataArray:
        """
        P(X >= threshold)
        """
        mu, sigma = self._get_mu_sigma(mean_da.values, spread_da.values)

        if not self.is_precip:
            z = (threshold - mu) / sigma
            prob = 1.0 - norm.cdf(z)
        else:
            w0, w1 = self.precip_logistic
            logits = w0 + w1 * mean_da.values
            prob_rain = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

            if threshold <= 0:
                prob = np.ones_like(mu)
            else:
                z = (np.sqrt(threshold) - mu) / sigma
                prob = prob_rain * (1.0 - norm.cdf(z))

        return xr.DataArray(prob, dims=mean_da.dims, coords=mean_da.coords)
