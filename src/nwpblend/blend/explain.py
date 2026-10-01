import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import xarray as xr


def get_global_importance(model, X_sample, M_sample, feature_names=None):
    """
    Computes global SHAP feature importance for the gating network.
    Returns mean absolute SHAP values per feature across all models.
    """
    model.eval()

    # SHAP DeepExplainer expects a background dataset
    background = X_sample[:100]
    M_sample[:100]

    # We need a wrapper to handle the mask input for SHAP
    # Since DeepExplainer takes a single tensor easily, we can bake the mask into the wrapper
    # Or just use GradientExplainer

    # For simplicity, we wrap the model to pass a fixed mask
    # (assuming we explain when all models are available)
    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model, mask):
            super().__init__()
            self.base_model = base_model
            self.mask = mask

        def forward(self, x):
            return self.base_model(x, self.mask[: x.shape[0]])

    wrapped = WrappedModel(model, M_sample)

    explainer = shap.DeepExplainer(wrapped, background)
    shap_values = explainer.shap_values(X_sample[:100])

    # shap_values is a list of arrays (one per output model)
    # We compute the mean absolute importance across all classes
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    if feature_names is not None:
        importance = {name: val for name, val in zip(feature_names, mean_abs_shap)}
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    return mean_abs_shap


def plot_weight_maps(
    weights_ds: xr.Dataset, var: str, date: str, lead: str, save_dir: str = "docs/figures"
):
    """
    Plots the spatial weight maps for a given date, lead, and variable.
    weights_ds: dataset containing the 'weights' DataArray with dims (time, lead, lat, lon, model)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select weights
    try:
        w_slice = weights_ds["weights"].sel(time=date, lead=lead)
    except KeyError:
        return

    models = w_slice.model.values

    _fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)

    for i, m in enumerate(models):
        ax = axes[0, i]
        w_m = w_slice.sel(model=m)
        im = ax.imshow(w_m.values, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{m} Weight")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Model Weights for {var} on {date} (Lead {lead})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"weights_map_{var}_{date}_{lead}.png"))
    plt.close()
