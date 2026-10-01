import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_qq_before_after(
    truth: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray, var_name: str, out_path: str
):
    """
    Plots Q-Q figure comparing raw vs truth and corrected vs truth.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Flatten and remove nans
    t = truth.values.flatten()
    r = raw.values.flatten()
    c = corrected.values.flatten()

    t = t[~np.isnan(t)]
    r = r[~np.isnan(r)]
    c = c[~np.isnan(c)]

    if len(t) == 0 or len(r) == 0:
        return

    quantiles = np.linspace(0.01, 0.99, 99)
    q_t = np.quantile(t, quantiles)
    q_r = np.quantile(r, quantiles)
    q_c = np.quantile(c, quantiles)

    plt.figure(figsize=(6, 6))
    plt.scatter(q_t, q_r, label="Raw", alpha=0.6)
    plt.scatter(q_t, q_c, label="Corrected", alpha=0.6)

    min_val = min(q_t.min(), q_r.min(), q_c.min())
    max_val = max(q_t.max(), q_r.max(), q_c.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Match")

    plt.xlabel("Truth Quantiles")
    plt.ylabel("Forecast Quantiles")
    plt.title(f"Q-Q Plot for {var_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
