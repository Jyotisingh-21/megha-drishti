import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class GatingNetwork(nn.Module):
    def __init__(self, in_features: int, n_models: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
        )

    def forward(self, x, mask):
        logits = self.mlp(x)
        # Mask unavailable models with very large negative number before softmax
        logits = logits.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        return weights


class GatingBlender:
    def __init__(self, n_models: int, is_precip: bool = False, model_dir: str = "models"):
        self.n_models = n_models
        self.is_precip = is_precip
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.feature_names = []

    def build_features(
        self,
        models_ds: xr.Dataset,
        truth_ds: xr.Dataset,
        avail_da: xr.DataArray,
        regime_probs: xr.DataArray,
        var: str,
    ):
        """
        Builds feature matrix for the Gating Network.
        regime_probs: (time, lat, lon, n_clusters)
        Returns: X (features), Y (truth), M (availability mask), F (forecasts matrix)
        """
        times = pd.to_datetime(models_ds.time.values)
        doy = times.dayofyear
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)

        lats = models_ds.lat.values
        lons = models_ds.lon.values
        lon2d, lat2d = np.meshgrid(lons, lats)

        sin_lat = np.sin(np.deg2rad(lat2d))
        cos_lat = np.cos(np.deg2rad(lat2d))
        sin_lon = np.sin(np.deg2rad(lon2d))
        cos_lon = np.cos(np.deg2rad(lon2d))

        leads = models_ds.lead.values.astype("timedelta64[h]").astype(float)

        n_clusters = regime_probs.shape[-1]

        X_list, Y_list, M_list, F_list = [], [], [], []

        # We will loop over time and lead to keep memory manageable
        for t_idx, t in enumerate(times):
            # Calculate trailing 7-day MAE
            t_dt = pd.to_datetime(t)
            t_start = t_dt - pd.Timedelta(days=7)
            t_end = t_dt - pd.Timedelta(days=1)

            try:
                hist_models = models_ds.sel(time=slice(t_start, t_end))
                hist_truth = truth_ds.sel(time=slice(t_start, t_end))
                err = np.abs(hist_models[var] - hist_truth[var])
                trailing_mae = err.mean(dim="time").transpose("lead", "lat", "lon", "model").values
            except (KeyError, ValueError):
                # If no history, default to 0
                trailing_mae = np.zeros((len(leads), len(lats), len(lons), self.n_models))

            avail_t = avail_da.sel(time=t).values  # (model)
            if not np.any(avail_t):
                continue

            for l_idx, lead in enumerate(leads):
                lead_norm = lead / 240.0

                # Spatial flat arrays
                n_pts = len(lats) * len(lons)

                # Base spatial/temporal
                feat = np.zeros((n_pts, 4 + 2 + 1 + n_clusters + self.n_models * 2))
                col = 0

                feat[:, col] = sin_lat.flatten()
                col += 1
                feat[:, col] = cos_lat.flatten()
                col += 1
                feat[:, col] = sin_lon.flatten()
                col += 1
                feat[:, col] = cos_lon.flatten()
                col += 1

                feat[:, col] = sin_doy[t_idx]
                col += 1
                feat[:, col] = cos_doy[t_idx]
                col += 1

                feat[:, col] = lead_norm
                col += 1

                # Regimes
                # regime_probs (time, lat, lon, cluster)
                # But our tag_day gives (1, cluster) which we broadcast, or we assume it's flat
                rp = regime_probs.sel(time=t).values  # (cluster) or (lat, lon, cluster)
                if rp.ndim == 1:
                    rp = np.broadcast_to(rp, (n_pts, n_clusters))
                else:
                    rp = rp.reshape(n_pts, n_clusters)

                feat[:, col : col + n_clusters] = rp
                col += n_clusters

                # Forecasts and trailing MAE
                fcsts = models_ds[var].isel(time=t_idx, lead=l_idx).values  # (lat, lon, model)
                fcsts_flat = fcsts.reshape(n_pts, self.n_models)
                feat[:, col : col + self.n_models] = fcsts_flat
                col += self.n_models

                tmae_flat = trailing_mae[l_idx].reshape(n_pts, self.n_models)
                feat[:, col : col + self.n_models] = tmae_flat
                col += self.n_models

                # Fill missing forecasts (from masked models) with 0
                feat = np.nan_to_num(feat, nan=0.0)

                mask_matrix = np.broadcast_to(avail_t, (n_pts, self.n_models))

                if truth_ds is not None:
                    truth_flat = truth_ds[var].isel(time=t_idx).values.flatten()
                    Y_list.append(truth_flat)

                X_list.append(feat)
                M_list.append(mask_matrix)
                F_list.append(np.nan_to_num(fcsts_flat, nan=0.0))

        if len(X_list) == 0:
            return None, None, None, None

        X = np.concatenate(X_list, axis=0)
        M = np.concatenate(M_list, axis=0)
        F = np.concatenate(F_list, axis=0)

        if truth_ds is not None:
            Y = np.concatenate(Y_list, axis=0)
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor(M, dtype=torch.float32),
                torch.tensor(F, dtype=torch.float32),
            )

        return (
            torch.tensor(X, dtype=torch.float32),
            None,
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(F, dtype=torch.float32),
        )

    def _loss_fn(self, weights, fcsts, truth):
        # weights: (batch, models)
        # fcsts: (batch, models)
        # truth: (batch,)
        pred = torch.sum(weights * fcsts, dim=1)
        err = (pred - truth) ** 2

        if self.is_precip:
            # Upweight heavy rain truth > 10mm
            weight_factor = 1.0 + (torch.clamp(truth, min=0.0) / 10.0)
            return torch.mean(err * weight_factor)
        else:
            return torch.mean(err)

    def fit(self, X, Y, M, F, epochs=10, batch_size=1024, lr=0.01):
        self.model = GatingNetwork(in_features=X.shape[1], n_models=self.n_models)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"X: {X.shape}, Y: {Y.shape}, M: {M.shape}, F: {F.shape}")
        dataset = TensorDataset(X, Y, M, F)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for ep in range(epochs):
            total_loss = 0
            for bx, by, bm, bf in loader:
                optimizer.zero_grad()
                w = self.model(bx, bm)
                loss = (
                    self.loss_fn(w, bf, by)
                    if hasattr(self, "loss_fn")
                    else self._loss_fn(w, bf, by)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {ep + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def predict_weights(self, X, M):
        self.model.eval()
        with torch.no_grad():
            w = self.model(X, M)
        return w.numpy()

    def save(self, name: str):
        path = os.path.join(self.model_dir, f"{name}.pt")
        torch.save(self.model.state_dict(), path)

    def load(self, name: str, in_features: int):
        path = os.path.join(self.model_dir, f"{name}.pt")
        self.model = GatingNetwork(in_features=in_features, n_models=self.n_models)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()


class LightGBMBaseline:
    """
    LightGBM stacking baseline.
    Predicts truth directly from the same features + forecasts.
    """

    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=50, max_depth=5, random_state=42)

    def fit(self, X_numpy, Y_numpy):
        self.model.fit(X_numpy, Y_numpy)

    def predict(self, X_numpy):
        return self.model.predict(X_numpy)
