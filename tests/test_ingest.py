import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.nwpblend.ingest.ecmwf import _process_file


@pytest.fixture
def dummy_grib_0p25(tmp_path):
    # Create a dummy xarray dataset resembling 0.25 ECMWF
    lats = np.arange(90, -90.01, -0.25)
    lons = np.arange(0, 360, 0.25)
    
    ds = xr.Dataset(
        {"2t": (("latitude", "longitude"), np.random.rand(len(lats), len(lons)) * 10 + 273.15)},
        coords={"latitude": lats, "longitude": lons, "step": pd.Timedelta(days=1)}
    )
    # mock to bypass cfgrib
    return ds

@pytest.fixture
def dummy_grib_0p1(tmp_path):
    # ECMWF 0.1 degree layout (Oct 2026 change)
    lats = np.arange(90, -90.01, -0.1)
    lons = np.arange(0, 360, 0.1)
    
    ds = xr.Dataset(
        # Some new files might use t2m directly
        {"t2m": (("latitude", "longitude"), np.random.rand(len(lats), len(lons)) * 10 + 273.15)},
        coords={"latitude": lats, "longitude": lons, "step": pd.Timedelta(days=1)}
    )
    return ds

def test_ecmwf_process_0p25(monkeypatch, dummy_grib_0p25):
    # Monkeypatch open_dataset
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: dummy_grib_0p25)
    
    domain = {"lat_min": 10, "lat_max": 20, "lon_min": 70, "lon_max": 80, "resolution": 0.25}
    ds_out = _process_file("fake.grib", domain, "2024-01-01")
    
    assert ds_out is not None
    assert "t2m" in ds_out
    assert ds_out["t2m"].attrs["units"] == "C"
    # Domain sizes: 20 down to 10 step -0.25 => 41 points
    assert len(ds_out.lat) == 41
    assert len(ds_out.lon) == 41
    assert ds_out.lat.values[0] == 20.0

def test_ecmwf_process_0p1(monkeypatch, dummy_grib_0p1):
    # Monkeypatch open_dataset
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: dummy_grib_0p1)
    
    domain = {"lat_min": 10, "lat_max": 20, "lon_min": 70, "lon_max": 80, "resolution": 0.25}
    ds_out = _process_file("fake.grib", domain, "2024-01-01")
    
    assert ds_out is not None
    assert "t2m" in ds_out
    assert ds_out["t2m"].attrs["units"] == "C"
    # Even if input is 0.1, output should be Regridded to 0.25
    assert len(ds_out.lat) == 41
    assert len(ds_out.lon) == 41
    assert ds_out.lat.values[0] == 20.0
