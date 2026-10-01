import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.ingest import ecmwf, gfs, ncmrwf_adapter


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.domain = {"lat_min": 10, "lat_max": 20, "lon_min": 70, "lon_max": 80}
        self.leads = [24, 48]
        self.variables = ["t2m", "precip"]
        self.date = "2024-07-30"

    @patch("ecmwf.opendata.Client")
    @patch("nwpblend.ingest.ecmwf.xr.open_dataset")
    @patch("nwpblend.ingest.ecmwf.get_cache_dir")
    def test_ecmwf_fetch(self, mock_get_cache_dir, mock_open_ds, mock_client):
        with tempfile.TemporaryDirectory() as d:
            mock_get_cache_dir.return_value = d

            # Create a dummy file to bypass the download check
            target_file = os.path.join(d, "ecmwf_2024-07-30.grib")
            with open(target_file, "w") as f:
                f.write("dummy")

            # Create a mock dataset
            mock_ds = xr.Dataset(
                {
                    "2t": (["step", "latitude", "longitude"], np.ones((2, 10, 10)) * 300.15),
                    "tp": (["step", "latitude", "longitude"], np.ones((2, 10, 10)) * 0.05),
                },
                coords={
                    "step": pd.to_timedelta([24, 48], unit="h"),
                    "latitude": np.linspace(20, 10, 10),
                    "longitude": np.linspace(70, 80, 10),
                },
            )
            mock_open_ds.return_value = mock_ds

            ds = ecmwf.fetch(self.date, self.domain, self.leads, self.variables)

            self.assertIsNotNone(ds)
            self.assertIn("t2m", ds)
            self.assertIn("precip", ds)
            # Check unit conversion
            self.assertAlmostEqual(ds["t2m"].values.flatten()[0], 27.0)  # 300.15 - 273.15
            self.assertAlmostEqual(ds["precip"].values.flatten()[0], 50.0)  # 0.05 * 1000

    @patch("s3fs.S3FileSystem")
    @patch("nwpblend.ingest.gfs.xr.open_dataset")
    @patch("nwpblend.ingest.gfs.get_cache_dir")
    def test_gfs_fetch(self, mock_get_cache_dir, mock_open_ds, mock_s3fs):
        with tempfile.TemporaryDirectory() as d:
            mock_get_cache_dir.return_value = d

            # Create a dummy file for each lead
            for lead in self.leads:
                target_file = os.path.join(d, f"gfs_20240730_f{lead:03d}.grib")
                with open(target_file, "w") as f:
                    f.write("dummy")

            mock_ds = xr.Dataset(
                {
                    "t2m": (["step", "latitude", "longitude"], np.ones((1, 10, 10)) * 300.15),
                    "prate": (["step", "latitude", "longitude"], np.ones((1, 10, 10)) * 0.0001),
                },
                coords={
                    "step": [pd.to_timedelta(24, unit="h")],
                    "latitude": np.linspace(20, 10, 10),
                    "longitude": np.linspace(70, 80, 10),
                },
            )
            mock_open_ds.return_value = mock_ds

            ds = gfs.fetch(self.date, self.domain, self.leads, self.variables)

            self.assertIsNotNone(ds)
            self.assertIn("t2m", ds)
            self.assertIn("precip", ds)

    def test_ncmrwf_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            ds = ncmrwf_adapter.fetch(
                self.date, self.domain, self.leads, self.variables, data_dir=d
            )
            self.assertIsNone(ds)
