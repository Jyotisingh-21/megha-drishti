import glob
import os
import shutil
import sys

import pytest
from streamlit.testing.v1 import AppTest

REPO_DIR = os.path.abspath(".")
sys.path.insert(0, os.path.join(REPO_DIR, "dashboard"))

PAGES = sorted([p.replace("\\", "/") for p in glob.glob("dashboard/pages/*.py")])

@pytest.fixture
def empty_processed_data(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "data" / "processed", exist_ok=True)
    os.makedirs(tmp_path / "data" / "demo" / "models", exist_ok=True)
    monkeypatch.chdir(tmp_path)
    os.makedirs("configs", exist_ok=True)
    with open("configs/thresholds.yaml", "w") as f:
        f.write("precip_mm_24h:\n  heavy: 64.5\nheatwave_celsius:\n  absolute:\n    plains: 40\n    hills: 30")


@pytest.fixture
def mock_demo_data(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "data" / "demo" / "models" / "ecmwf_ifs.zarr", exist_ok=True)
    import numpy as np
    import pandas as pd
    import xarray as xr
    time = pd.date_range("2024-01-01", periods=1)
    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], np.ones((1, 5, 5))),
            "precip": (["time", "lat", "lon"], np.ones((1, 5, 5))),
        },
        coords={"time": time, "lat": np.linspace(10, 20, 5), "lon": np.linspace(70, 80, 5)}
    )
    import dashboard.data_loader as dl
    monkeypatch.setattr(dl, "_load_data_cached", lambda dummy: (ds, True))
    monkeypatch.chdir(tmp_path)
    # mock threshold
    os.makedirs("configs", exist_ok=True)
    with open("configs/thresholds.yaml", "w") as f:
        f.write("precip_mm_24h:\n  heavy: 64.5\nheatwave_celsius:\n  absolute:\n    plains: 40\n    hills: 30")

@pytest.fixture
def mock_real_data(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr
    time = pd.date_range("2024-01-01", periods=1)
    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], np.ones((1, 5, 5))),
            "precip": (["time", "lat", "lon"], np.ones((1, 5, 5))),
        },
        coords={"time": time, "lat": np.linspace(10, 20, 5), "lon": np.linspace(70, 80, 5)}
    )
    import dashboard.data_loader as dl
    monkeypatch.setattr(dl, "_load_data_cached", lambda dummy: (ds, False))
    monkeypatch.chdir(tmp_path)
    os.makedirs("configs", exist_ok=True)
    with open("configs/thresholds.yaml", "w") as f:
        f.write("precip_mm_24h:\n  heavy: 64.5\nheatwave_celsius:\n  absolute:\n    plains: 40\n    hills: 30")

@pytest.mark.parametrize("page", PAGES)
def test_page_empty_data(empty_processed_data, page, monkeypatch):
    import dashboard.data_loader as dl
    monkeypatch.setattr(dl, "_load_data_cached", lambda dummy: (None, False))
    at = AppTest.from_file(os.path.join(REPO_DIR, "dashboard", "app.py"), default_timeout=20).run(timeout=20)
    at.switch_page(page.replace("dashboard/", "")).run(timeout=20)
    assert not at.exception, f"Page {page} crashed when data is missing: {at.exception[0].message if at.exception else ''}"
    if "about" not in page and "pipeline_status" not in page:
        assert any("No real forecast data yet" in str(getattr(info, "value", "")) for info in at.info), f"Page {page} missing graceful message"

@pytest.mark.parametrize("page", PAGES)
def test_page_demo_data(mock_demo_data, page):
    at = AppTest.from_file(os.path.join(REPO_DIR, "dashboard", "app.py"), default_timeout=20).run(timeout=20)
    at.switch_page(page.replace("dashboard/", "")).run(timeout=20)
    assert not at.exception, f"Page {page} crashed in DEMO mode: {at.exception[0].message if at.exception else ''}"

@pytest.mark.parametrize("page", PAGES)
def test_page_real_data(mock_real_data, page):
    at = AppTest.from_file(os.path.join(REPO_DIR, "dashboard", "app.py"), default_timeout=20).run(timeout=20)
    at.switch_page(page.replace("dashboard/", "")).run(timeout=20)
    assert not at.exception, f"Page {page} crashed in REAL mode: {at.exception[0].message if at.exception else ''}"
