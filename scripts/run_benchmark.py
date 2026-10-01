import argparse
import logging

import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.blend.baselines import bma, equal_weight, ewa
from nwpblend.verify.metrics import bias, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.demo:
        logger.info("Only --demo mode is supported for this run_benchmark script.")
        return

    logger.info("Loading demo data for benchmark...")
    try:
        truth = xr.open_zarr("data/demo/truth.zarr").load()
        import os

        if not os.path.exists("data/processed/stacked_models.zarr"):
            from nwpblend.harmonise.store import stack_models, write_store

            logger.info("Stacked models not found. Stacking demo data now...")
            expected = ["ecmwf_ifs", "gfs", "aifs", "ncum_g", "graphcast", "pangu"]
            m_dict = {}
            for m in expected:
                try:
                    m_dict[m] = xr.open_zarr(f"data/demo/models/{m}.zarr").load()
                except Exception:
                    logger.warning(f"Demo model {m} not found.")

            if m_dict:
                stacked = stack_models(m_dict, expected)
                os.makedirs("data/processed", exist_ok=True)
                write_store(stacked, "data/processed/stacked_models.zarr")

        models = xr.open_zarr("data/processed/stacked_models.zarr").load()
    except Exception as e:
        logger.error(f"Failed to load stacked models or truth. Ensure Prompt 1 is run. {e}")
        return

    # Define test period (last 30 days)
    times = models.time.values
    if len(times) <= 30:
        logger.error("Dataset too small for 30-day test window.")
        return

    test_times = times[-30:]
    truth_test = truth.sel(time=test_times)

    # Baselines (evaluates over test_times but requires full time for history)
    # To save memory, we can slice models
    models_test = models.sel(time=test_times)
    available_test = models.available.sel(time=test_times)

    logger.info("Computing equal weights...")
    eq_blend = equal_weight(models_test, available_test)

    logger.info("Computing EWA...")
    ewa_blend = ewa(models, truth, models.available, window=30).sel(time=test_times)

    logger.info("Computing BMA...")
    bma_blend = bma(models, truth, models.available, window=30).sel(time=test_times)

    print("=== Benchmark Results (DEMO Data) ===")

    # We will output a markdown table
    out_lines = ["| Model | Var | RMSE | MAE | Bias |", "|-------|-----|------|-----|------|"]

    def evaluate_dataset(name, ds, var, lead):
        # Flatten over lat/lon, average over time
        fcst_val = ds[var].sel(lead=lead)
        truth_val = truth_test[var]

        # Align
        if fcst_val.shape != truth_val.shape:
            return

        r = float(rmse(fcst_val, truth_val).mean().values)
        m = float(mae(fcst_val, truth_val).mean().values)
        b = float(bias(fcst_val, truth_val).mean().values)

        out_lines.append(f"| {name} | {var} | {r:.3f} | {m:.3f} | {b:.3f} |")

    lead = models.lead.values[0]
    variables = ["t2m", "precip"]

    for var in variables:
        for m_name in models.model.values:
            m_ds = models_test.sel(model=m_name)
            evaluate_dataset(f"Single: {m_name}", m_ds, var, lead)

        evaluate_dataset("Blend: Equal", eq_blend, var, lead)
        evaluate_dataset("Blend: EWA", ewa_blend, var, lead)
        evaluate_dataset("Blend: BMA", bma_blend, var, lead)

    for line in out_lines:
        print(line)

    with open("docs/results_demo.md", "w") as f:
        f.write("# Benchmark Results (DEMO Data)\n\n")
        f.write("Evaluation of single models vs baseline blenders.\n\n")
        f.write("\n".join(out_lines))
        f.write("\n")


if __name__ == "__main__":
    main()
