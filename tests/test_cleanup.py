import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nwpblend.pipeline import verify_and_save_run


def test_verify_and_save_run(tmp_path, monkeypatch):
    config = {"models": ["ecmwf_ifs"]}
    domain = {"resolution": 0.25}
    run_dt = datetime(2026, 1, 1, tzinfo=UTC)
    run_str = "20260101_0000"

    monkeypatch.chdir(tmp_path)
    os.makedirs(f"data/archive/ecmwf_ifs/{run_str}", exist_ok=True)
    raw_file = f"data/archive/ecmwf_ifs/{run_str}/raw.grib"
    with open(raw_file, "w") as f:
        f.write("mock")

    class MockIngest:
        @staticmethod
        def _process_file(f, dom, dt):
            dt_naive = pd.to_datetime(dt).tz_localize(None)
            return xr.Dataset(
                {"t2m": (("lead", "lat", "lon"), [[[1.0]]])},
                coords={"lead": [24], "lat": [10.0], "lon": [80.0], "time": [dt_naive]},
            )

    import nwpblend.ingest.ecmwf as ecmwf_ingest

    monkeypatch.setattr(ecmwf_ingest, "_process_file", MockIngest._process_file)

    def mock_fetch_truth(start, end, dom):
        times = pd.date_range(start, end, freq="D")
        return xr.Dataset(
            {"t2m": (("time", "lat", "lon"), np.ones((len(times), 1, 1)))},
            coords={"time": times, "lat": [10.0], "lon": [80.0]},
        )

    import nwpblend.truth

    monkeypatch.setattr(nwpblend.truth, "fetch_real_truth", mock_fetch_truth)

    res = verify_and_save_run(run_dt, config, domain)
    assert res is True
    assert os.path.exists(f"data/verified/pair_{run_str}/models.zarr")
