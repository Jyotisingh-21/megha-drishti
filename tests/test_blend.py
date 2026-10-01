import numpy as np
import xarray as xr

from nwpblend.blend.baselines import equal_weight


def test_equal_weight():
    # 2 models, 1 missing
    data = np.array([[1.0, np.nan], [2.0, 4.0]])  # shape (time=2, model=2)

    models = xr.Dataset(
        {"t2m": (["time", "model"], data)}, coords={"time": [1, 2], "model": ["m1", "m2"]}
    )

    available = xr.DataArray(
        [[True, False], [True, True]],
        dims=["time", "model"],
        coords={"time": [1, 2], "model": ["m1", "m2"]},
    )

    out = equal_weight(models, available)

    # Time 1: only m1 is available, so mean is 1.0
    # Time 2: m1 (2.0) and m2 (4.0) are available, so mean is 3.0
    assert np.allclose(out["t2m"].values, [1.0, 3.0])
