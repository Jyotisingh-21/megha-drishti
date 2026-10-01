import argparse
import logging

import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.biascorrect.core import BiasCorrector
from nwpblend.biascorrect.plot import plot_qq_before_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.demo:
        logger.info("Only --demo mode is supported for this report script.")
        return

    logger.info("Loading demo data...")
    try:
        truth = xr.open_zarr("data/demo/truth.zarr")
        ecmwf = xr.open_zarr("data/demo/models/ecmwf_ifs.zarr")
        gfs = xr.open_zarr("data/demo/models/gfs.zarr")
    except Exception as e:
        logger.error(f"Failed to load demo data. Run Prompt 1 script first. {e}")
        return

    models = {"ecmwf_ifs": ecmwf, "gfs": gfs}

    # Split into train/test (no leakage)
    # Total ~ 90 days. First 60 train, rest test.
    times = truth.time.values
    split_idx = 60
    if len(times) <= split_idx:
        split_idx = len(times) // 2

    train_times = times[:split_idx]
    test_times = times[split_idx:]

    truth_train = truth.sel(time=train_times)
    truth_test = truth.sel(time=test_times)

    print("=== Bias Correction Report (Demo Data) ===")
    print(
        f"{'Model':<15} | {'Var':<10} | {'RMSE (Raw)':<12} | {'RMSE (Corr)':<12} | {'Bias (Raw)':<12} | {'Bias (Corr)':<12}"
    )
    print("-" * 85)

    for name, ds in models.items():
        ds_train = ds.sel(time=train_times)
        ds_test = ds.sel(time=test_times)

        bc = BiasCorrector()
        bc.fit(ds_train, truth_train)

        ds_corr = bc.transform(ds_test)

        for var in ["t2m", "precip"]:
            # Evaluate on lead=24h (index 0 or search)
            lead = ds.lead.values[0]

            raw_val = ds_test[var].sel(lead=lead).values
            corr_val = ds_corr[var].sel(lead=lead).values
            truth_val = truth_test[var].values

            # Align shapes (truth has no lead)
            if raw_val.shape != truth_val.shape:
                continue

            rmse_raw = np.sqrt(np.nanmean((raw_val - truth_val) ** 2))
            rmse_corr = np.sqrt(np.nanmean((corr_val - truth_val) ** 2))

            bias_raw = np.nanmean(raw_val - truth_val)
            bias_corr = np.nanmean(corr_val - truth_val)

            print(
                f"{name:<15} | {var:<10} | {rmse_raw:<12.3f} | {rmse_corr:<12.3f} | {bias_raw:<12.3f} | {bias_corr:<12.3f}"
            )

            plot_qq_before_after(
                truth_test[var],
                ds_test[var].sel(lead=lead),
                ds_corr[var].sel(lead=lead),
                var,
                f"docs/figures/qq_{name}_{var}.png",
            )


if __name__ == "__main__":
    main()
