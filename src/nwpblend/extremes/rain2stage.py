import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm


class TwoStagePrecipModel:
    def __init__(self, random_state=42):
        self.clf = lgb.LGBMClassifier(n_estimators=50, random_state=random_state)
        # We use standard L2 loss on log(y) for the amount model
        self.reg = lgb.LGBMRegressor(n_estimators=50, random_state=random_state)
        self.is_fitted = False

    def _prepare_features(self, X_da: xr.Dataset, var_name="precip"):
        """
        Flattens the spatial dataset to feature matrix for LightGBM.
        """
        # Assume X_da is output from baselines or stacked models
        # For simplicity, we just use the raw mean and spread, or individual members if present

        # If it's the stacked models dataset
        if "model" in X_da.dims:
            # use mean across available models
            mean_val = X_da[var_name].mean(dim="model").values.flatten()
            spread_val = X_da[var_name].std(dim="model").values.flatten()
        else:
            mean_val = X_da[var_name].values.flatten()
            spread_val = np.zeros_like(mean_val)

        # Add spatial/temporal feats
        n_pts = len(mean_val)
        X = np.zeros((n_pts, 2))
        X[:, 0] = mean_val
        X[:, 1] = spread_val
        return X

    def fit(self, X_ds: xr.Dataset, truth_da: xr.DataArray, var_name="precip"):
        X = self._prepare_features(X_ds, var_name)
        Y = truth_da.values.flatten()

        # Remove NaNs
        mask = ~np.isnan(X[:, 0]) & ~np.isnan(Y)
        X = X[mask]
        Y = Y[mask]

        # 1. Classifier: rain / no-rain (> 0.1mm)
        rain_mask = Y > 0.1
        self.clf.fit(X, rain_mask.astype(int))

        # 2. Regressor: amount conditional on rain
        if np.sum(rain_mask) > 10:
            X_wet = X[rain_mask]
            Y_wet = Y[rain_mask]

            # log(y) target
            Y_wet_log = np.log(np.clip(Y_wet, 1e-6, None))

            # Event-weighted (upweight heavy rain)
            weights = 1.0 + (Y_wet / 10.0)

            self.reg.fit(X_wet, Y_wet_log, sample_weight=weights)

        self.is_fitted = True
        return self

    def predict_distribution(self, X_ds: xr.Dataset, var_name="precip") -> xr.Dataset:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X = self._prepare_features(X_ds, var_name)

        # 1. PoP
        pop = self.clf.predict_proba(X)[:, 1]

        # 2. Amount
        pred_log = self.reg.predict(X)
        amount = np.exp(pred_log)

        # Expected value
        expected = pop * amount

        # Reshape to original dims
        if "model" in X_ds.dims:
            out_dims = [d for d in X_ds[var_name].dims if d != "model"]
            coords = {d: X_ds[d] for d in out_dims}
        else:
            out_dims = X_ds[var_name].dims
            coords = X_ds.coords

        shape = tuple(len(coords[d]) for d in out_dims)

        ds = xr.Dataset(
            {
                "pop": (out_dims, pop.reshape(shape)),
                "conditional_amount": (out_dims, amount.reshape(shape)),
                "expected_amount": (out_dims, expected.reshape(shape)),
            },
            coords=coords,
        )
        return ds

    def exceedance_prob(
        self, threshold: float, X_ds: xr.Dataset, var_name="precip"
    ) -> xr.DataArray:
        """
        Very crude exceedance probability since stage 2 is deterministic (point prediction).
        We approximate the stage 2 variance to construct a probability, or just use indicator.
        For a full probabilistic model, stage 2 should predict parameters of Gamma.
        Here we use a rough log-normal spread proxy based on GBM tree variance if possible,
        or a fixed scale for demonstration.
        """
        ds = self.predict_distribution(X_ds, var_name)
        pop = ds["pop"]
        mu = np.log(np.clip(ds["conditional_amount"], 1e-6, None))

        # Mock variance of log-normal for demo (0.5 std dev on log scale)
        sigma = 0.5

        if threshold <= 0.1:
            return pop

        z = (np.log(threshold) - mu) / sigma
        # P(amount > thresh | rain) = 1 - norm.cdf(z)
        p_cond = 1.0 - norm.cdf(z)

        p_exceed = pop * p_cond
        return xr.DataArray(p_exceed, dims=ds.dims, coords=ds.coords)
