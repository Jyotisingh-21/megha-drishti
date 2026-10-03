import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.harmonise.store import stack_models


def main():
    time = pd.date_range("2024-01-01", periods=1)
    lat = np.array([10.0, 11.0])
    lon = np.array([77.0, 78.0])

    dsA = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.random.rand(1, 3, 2, 2))},
        coords={"time": time, "lead": [24, 48, 72], "lat": lat, "lon": lon},
    )

    dsB = xr.Dataset(
        {"t2m": (("time", "lead", "lat", "lon"), np.random.rand(1, 1, 2, 2))},
        coords={"time": time, "lead": [24], "lat": lat, "lon": lon},
    )

    models = {"ecmwf": dsA, "gfs": dsB}
    expected = ["ecmwf", "gfs", "ncum"]

    try:
        stacked = stack_models(models, expected)
        print(stacked.dims)
        print("Available shape:", stacked["available"].shape)
        print("Available models at lead 72:", stacked["available"].sel(lead=72).values)
        print("Available models at lead 24:", stacked["available"].sel(lead=24).values)
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
