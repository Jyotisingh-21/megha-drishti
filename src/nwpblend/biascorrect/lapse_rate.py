import logging

import xarray as xr

logger = logging.getLogger(__name__)


def correct_t2m(
    ds: xr.Dataset,
    oro_model: xr.DataArray = None,
    oro_truth: xr.DataArray = None,
    lapse_rate: float = 0.0065,
) -> xr.Dataset:
    """
    Corrects 2m temperature for elevation mismatches using a constant lapse rate.

    Args:
        ds: Dataset containing 't2m'.
        oro_model: Model orography in meters.
        oro_truth: Truth orography in meters.
        lapse_rate: Lapse rate in K/m (default 6.5 K/km).

    Returns:
        Corrected Dataset.
    """
    if "t2m" not in ds.data_vars:
        return ds

    if oro_model is None or oro_truth is None:
        logger.warning("Orography missing. Skipping lapse rate correction for t2m.")
        return ds

    # Temperature decreases with height.
    # If model elevation > truth elevation, model is too cold, we must add temp.
    # T_corr = T_model + lapse_rate * (H_model - H_truth)

    out = ds.copy()
    diff = oro_model - oro_truth

    out["t2m"] = ds["t2m"] + (lapse_rate * diff)
    out["t2m"].attrs.update(ds["t2m"].attrs)

    return out
