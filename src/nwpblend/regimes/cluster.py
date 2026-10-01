import logging
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeClusterer:
    def __init__(self, n_clusters: int = 4, n_pca_components: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _prepare_features(self, ds: xr.Dataset) -> np.ndarray:
        """
        Flatten spatial dimensions and concatenate variables to form feature vectors per time step.
        """
        features = []
        for var in ds.data_vars:
            # Flatten lat/lon
            # array shape: (time, lat, lon) -> (time, lat*lon)
            arr = ds[var].values
            if arr.ndim == 3:
                flat = arr.reshape(arr.shape[0], -1)
                features.append(flat)
            elif arr.ndim == 4:
                # E.g. has lead time, we'd take lead=0 or mean
                flat = arr[:, 0, :, :].reshape(arr.shape[0], -1)
                features.append(flat)

        if not features:
            return np.array([])

        return np.concatenate(features, axis=1)

    def fit(self, ds: xr.Dataset):
        X = self._prepare_features(ds)
        if len(X) == 0:
            return

        X_scaled = self.scaler.fit_transform(X)

        # In case n_samples < n_pca_components
        self.pca.n_components = min(self.pca.n_components, X_scaled.shape[0])

        X_pca = self.pca.fit_transform(X_scaled)
        self.kmeans.fit(X_pca)
        self._is_fitted = True
        return self

    def predict(self, ds: xr.Dataset) -> xr.DataArray:
        if not self._is_fitted:
            raise ValueError("Clusterer is not fitted.")

        X = self._prepare_features(ds)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        labels = self.kmeans.predict(X_pca)

        return xr.DataArray(labels, dims=["time"], coords={"time": ds.time}, name="regime_id")

    def tag_day(self, ds_day: xr.Dataset):
        """
        Returns regime_id and soft probabilities for a single day (or set of days).
        Soft probabilities computed via distance to centroids (softmax).
        """
        X = self._prepare_features(ds_day)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        labels = self.kmeans.predict(X_pca)

        # Soft probabilities based on negative distance
        distances = self.kmeans.transform(X_pca)
        # Softmax over negative squared distances (or just inverse distance)
        exp_d = np.exp(-distances)
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)

        return labels, probs

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pca": self.pca,
                    "kmeans": self.kmeans,
                    "scaler": self.scaler,
                    "n_clusters": self.n_clusters,
                    "is_fitted": self._is_fitted,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.pca = state["pca"]
            self.kmeans = state["kmeans"]
            self.scaler = state["scaler"]
            self.n_clusters = state["n_clusters"]
            self._is_fitted = state["is_fitted"]
        return self
