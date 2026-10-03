import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nwpblend.pipeline import run_daily


@pytest.fixture
def mock_ingest(monkeypatch):
    """Mocks ecmwf and gfs to return different leads."""

    def mock_ecmwf(date, domain, leads, variables):
        time = pd.date_range("2024-01-01", periods=1)
        lat = np.array([10.0, 11.0])
        lon = np.array([77.0, 78.0])
        # Returns ALL leads
        ds = xr.Dataset(
            {
                "t2m": (("time", "lead", "lat", "lon"), np.ones((1, len(leads), 2, 2))),
                "precip": (("time", "lead", "lat", "lon"), np.ones((1, len(leads), 2, 2))),
            },
            coords={"time": time, "lead": leads, "lat": lat, "lon": lon},
        )
        return ds, "2024-01-01 00:00"

    def mock_gfs(date, domain, leads, variables):
        time = pd.date_range("2024-01-01", periods=1)
        lat = np.array([10.0, 11.0])
        lon = np.array([77.0, 78.0])
        # Returns ONLY the first lead
        partial_leads = [leads[0]]
        ds = xr.Dataset(
            {
                "t2m": (("time", "lead", "lat", "lon"), np.ones((1, 1, 2, 2))),
                "precip": (("time", "lead", "lat", "lon"), np.ones((1, 1, 2, 2))),
            },
            coords={"time": time, "lead": partial_leads, "lat": lat, "lon": lon},
        )
        return ds, "2024-01-01 00:00"

    monkeypatch.setattr("nwpblend.ingest.ecmwf.probe_and_fetch", mock_ecmwf)
    monkeypatch.setattr("nwpblend.ingest.gfs.probe_and_fetch", mock_gfs)


def test_partial_leads_end_to_end(mock_ingest, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    os.makedirs("configs", exist_ok=True)
    with open("configs/default.yaml", "w") as f:
        f.write("""
models:
  - ecmwf_ifs
  - gfs
variables:
  - t2m
  - precip
lead_times_days: [1, 2, 3]
""")
    with open("configs/thresholds.yaml", "w") as f:
        f.write("""
precip_mm_24h:
  heavy: 64.5
""")

    # Run the pipeline (skip_download=False so it hits the mocks)
    domain = {"lat_max": 20, "lat_min": 10, "lon_min": 70, "lon_max": 80}
    report = run_daily("2024-01-01", domain, demo=False, skip_download=False, quick=True)

    assert report["status"] == "SUCCESS", "Pipeline should succeed despite partial data"
    assert report["stages"]["ingest"]["status"] == "PARTIAL", "Ingest should be marked PARTIAL"
    assert (
        "GFS partial run. Missing leads: [48, 72]" in report["warnings"][-1]
        or "gfs partial run" in report["warnings"][-1].lower()
        or "missing leads: [48, 72]" in str(report["warnings"])
    )

    # Verify the final export has all leads
    assert os.path.exists("data/output/blend_2024-01-01.nc")
    out = xr.open_dataset("data/output/blend_2024-01-01.nc")
    assert list(out.lead.values) == [24, 48, 72]

    # Check the report
    with open("data/logs/run_report_latest.json", "r") as f:
        rep = json.load(f)
        assert rep["stages"]["ingest"]["status"] == "PARTIAL"
        assert "gfs: PARTIAL" in rep["stages"]["ingest"]["details"]
        assert rep["stages"]["ingest"]["status"] == "PARTIAL"
        assert "gfs: PARTIAL" in rep["stages"]["ingest"]["details"]
