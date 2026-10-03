import numpy as np
import pytest
import xarray as xr

from nwpblend.ingest.regrid import regrid_dataset


def test_regrid_dataset_conservative_precip():
    # 0.1 deg input
    lats = np.arange(38, 7.9, -0.1)
    lons = np.arange(68, 98.1, 0.1)
    # Precip is 1.0 everywhere
    data = np.ones((len(lats), len(lons)))
    
    ds = xr.Dataset(
        {
            "precip": (["lat", "lon"], data),
            "t2m": (["lat", "lon"], data * 2.0)
        },
        coords={"lat": lats, "lon": lons}
    )
    
    domain = {
        "lat_min": 8.0,
        "lat_max": 38.0,
        "lon_min": 68.0,
        "lon_max": 98.0,
        "resolution": 0.25
    }
    
    original_mean_precip = float(ds["precip"].mean())
    
    out = regrid_dataset(ds, domain)
    
    assert "precip" in out
    assert "t2m" in out
    assert len(out.lat) == 121  # (38 - 8) / 0.25 + 1
    assert len(out.lon) == 121
    
    new_mean_precip = float(out["precip"].mean())
    
    # Should be conserved
    np.testing.assert_allclose(original_mean_precip, new_mean_precip, rtol=1e-3)
    np.testing.assert_allclose(float(out["t2m"].mean()), 2.0, rtol=1e-3)
