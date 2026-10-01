import logging
import os
from datetime import UTC, datetime, timezone
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)


def export_netcdf(ds: xr.Dataset, out_path: str, metadata: dict[str, Any] | None = None):
    """
    Exports Dataset to CF-compliant NetCDF4 with compression.
    """
    ds_out = ds.copy()

    # Global Attributes
    ds_out.attrs["Conventions"] = "CF-1.8"
    ds_out.attrs["title"] = "Megha-Drishti Blended Forecast"
    ds_out.attrs["history"] = f"Created at {datetime.now(UTC).isoformat()}Z"

    if metadata:
        for k, v in metadata.items():
            ds_out.attrs[k] = str(v)

    # Variable Attributes
    for var in ds_out.data_vars:
        if var == "precip":
            ds_out[var].attrs["units"] = "mm/day"
            ds_out[var].attrs["long_name"] = "24-hour precipitation accumulation"
        elif var == "t2m":
            ds_out[var].attrs["units"] = "Celsius"
            ds_out[var].attrs["long_name"] = "2-metre temperature"
        elif var == "wind10m":
            ds_out[var].attrs["units"] = "m/s"
            ds_out[var].attrs["long_name"] = "10-metre wind speed"

    # Compression
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds_out.to_netcdf(out_path, format="NETCDF4", engine="netcdf4", encoding=encoding)
    logger.info(f"Exported NetCDF to {out_path}")


def export_grib2(ds: xr.Dataset, out_path: str):
    """
    Exports to GRIB2 using eccodes if available.
    """
    try:
        import eccodes

        # Writing GRIB2 from xarray using python bindings is highly involved.
        # This is a stub for where the translation logic goes.
        logger.info(f"eccodes is available. Writing GRIB2 to {out_path} (Stubbed)")
        # stub implementation
        with open(out_path, "wb") as f:
            f.write(b"GRIB2 STUB DATA")
    except ImportError:
        logger.warning(
            "eccodes library not found. Skipping GRIB2 export. Falling back to NetCDF only."
        )
