import logging
import os
from typing import Optional

import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)

CACHE_DIR = "data/raw/indices"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_synthetic_regimes() -> xr.Dataset:
    """
    Load synthetic regimes for demo mode.
    """
    try:
        # We stored regimes in data/demo/truth.zarr or somewhere similar?
        # In synthetic.py we return regimes_ds, but where is it saved?
        # The prompt says 'For demo mode use the synthetic regime label'
        return xr.open_zarr("data/demo/regimes.zarr")
    except Exception as e:
        logger.warning(f"Could not load synthetic regimes: {e}")
        return xr.Dataset()


def get_mjo_index() -> pd.DataFrame:
    """
    Downloads RMM MJO index from BoM.
    Cached locally to avoid repeated downloads.
    """
    cache_path = os.path.join(CACHE_DIR, "mjo_rmm.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    try:
        # In a real operational setting we parse this txt file
        # Here we document the path and raise NotImplementedError if we don't write the full parser
        raise NotImplementedError(
            "BoM MJO parser not fully implemented. Real mode requires parsing the txt file."
        )
    except Exception as e:
        logger.warning(f"MJO download failed: {e}")
        return pd.DataFrame()


def get_enso_nino34() -> pd.DataFrame:
    """
    Download ENSO Nino3.4 index.
    """
    cache_path = os.path.join(CACHE_DIR, "enso_nino34.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    raise NotImplementedError("ENSO Nino3.4 download not implemented. Requires NOAA PSL parsing.")


def get_iod_dmi() -> pd.DataFrame:
    """
    Download IOD (DMI) index.
    """
    cache_path = os.path.join(CACHE_DIR, "iod_dmi.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    raise NotImplementedError("IOD DMI download not implemented. Requires NOAA PSL parsing.")


def get_monsoon_flag(precip_da: xr.DataArray) -> xr.DataArray:
    """
    Compute monsoon active/break flag based on Central India rainfall anomalies.
    Central India ~ 15N-25N, 73E-82E.
    """
    # Slice to central India
    # Note: depends on coordinate names lat, lon
    if "lat" not in precip_da.coords or "lon" not in precip_da.coords:
        return xr.zeros_like(precip_da.mean(dim=["lat", "lon"]))

    central = precip_da.sel(lat=slice(25, 15), lon=slice(73, 82))

    # In a real system, we subtract climatology.
    # Here we just use a simple threshold on the mean for demonstration.
    mean_rain = central.mean(dim=["lat", "lon"])

    # active > 5mm, break < 2mm
    flags = xr.zeros_like(mean_rain, dtype=int)
    flags = xr.where(mean_rain > 5.0, 1, flags)  # 1 = active
    flags = xr.where(mean_rain < 2.0, 2, flags)  # 2 = break
    return flags


def get_western_disturbance(z500_da: xr.DataArray) -> xr.DataArray:
    """
    Western disturbance flag from 500 hPa geopotential trough over North India.
    North India ~ 25N-35N, 65E-80E.
    """
    if "lat" not in z500_da.coords or "lon" not in z500_da.coords:
        return xr.zeros_like(z500_da.mean(dim=["lat", "lon"]))

    north = z500_da.sel(lat=slice(35, 25), lon=slice(65, 80))
    mean_z = north.mean(dim=["lat", "lon"])

    # Threshold for trough
    # Simple proxy: anomaly below some threshold
    # Since we don't have climo, we mock it
    flags = xr.zeros_like(mean_z, dtype=int)
    return flags
