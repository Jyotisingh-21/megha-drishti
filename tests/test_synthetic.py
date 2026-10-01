import numpy as np

from nwpblend.ingest.synthetic import generate_synthetic_data


def test_synthetic_schema():
    truth, models, ensembles, _regimes = generate_synthetic_data(
        lat_min=10, lat_max=15, lon_min=70, lon_max=75, resolution=5, days=5
    )

    # 1. Dimensions and Coordinate checks
    assert "time" in truth.coords
    assert "lat" in truth.coords
    assert "lon" in truth.coords

    m_ds = models["gfs"]
    assert "lead" in m_ds.coords

    ens_ds = ensembles["neps"]
    assert "member" in ens_ds.coords
    assert len(ens_ds.member) == 10

    # 2. Variable and Unit checks
    for var in ["precip", "t2m", "wind10m", "gust10m"]:
        assert var in truth
        assert var in m_ds
        assert "units" in m_ds[var].attrs

    # 3. No NaNs where not expected
    assert not m_ds["t2m"].isnull().any()

    # 4. Model differences by lead and model
    ai = models["graphcast"]["t2m"].values
    nwp = models["gfs"]["t2m"].values
    truth_val = truth["t2m"].values

    ai_rmse_lead1 = np.sqrt(np.mean((ai[:, 0, :, :] - truth_val) ** 2))
    ai_rmse_lead10 = np.sqrt(np.mean((ai[:, -1, :, :] - truth_val) ** 2))

    # RMSE should grow with lead
    assert ai_rmse_lead10 > ai_rmse_lead1

    nwp_rmse_lead10 = np.sqrt(np.mean((nwp[:, -1, :, :] - truth_val) ** 2))

    # We built AI to have lower temperature error at long leads than NWP
    assert ai_rmse_lead10 < nwp_rmse_lead10
