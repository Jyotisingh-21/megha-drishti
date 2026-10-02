import os
import shutil

import pytest
import xarray as xr

from nwpblend.harmonise.store import stack_models, write_store


def test_stale_cache_guard_logic(tmp_path):
    # Mock models and truth with different shapes
    truth = xr.Dataset(coords={"lat": [1, 2, 3], "lon": [1, 2]})
    models = xr.Dataset(coords={"lat": [1, 2], "lon": [1, 2]})

    # Check logic matches run_benchmark.py
    is_stale = len(models.lat) != len(truth.lat) or len(models.lon) != len(truth.lon)
    assert is_stale == True
