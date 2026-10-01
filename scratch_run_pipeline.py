import pandas as pd
from nwpblend.pipeline import build_dataset

# We need dates for the demo data. In synthetic.py it's 90 days from 2024-07-30?
# Wait, synthetic.py generated ~90 days starting from what date? Let's check download_demo_data.py or truth.zarr
import xarray as xr
truth = xr.open_zarr("data/demo/truth.zarr")
dates = pd.to_datetime(truth.time.values).strftime('%Y-%m-%d').tolist()

# The models expected in the stacked store
expected_models = ["ecmwf_ifs", "gfs", "aifs", "ncum_g", "graphcast", "pangu"]

build_dataset(dates, expected_models)
