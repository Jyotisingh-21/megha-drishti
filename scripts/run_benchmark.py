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

        # STALE CACHE GUARD: Check if dimensions match between truth and models
        if len(models.lat) != len(truth.lat) or len(models.lon) != len(truth.lon):
            logger.warning("Stale cache detected: 'lat' or 'lon' sizes mismatch between truth and models! Rebuilding stacked models...")
            import shutil
            shutil.rmtree("data/processed/stacked_models.zarr")
            
            # Restack
            stacked = stack_models(m_dict, expected)
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
        f.write("\n\n")

    # Run Ablation Study
    logger.info("Running Ablation Study...")
    try:
        from nwpblend.verify.ablation import run_ablation

        ablation_df = run_ablation(models, truth, models.available)

        # Run Replay
        logger.info("Running Event Replay...")
        from nwpblend.verify.replay import check_model_drift, replay_events

        replay_report = replay_events(models, truth, is_demo=True)
        drift_flags = check_model_drift(models, truth, models.available)

        with open("docs/results_demo.md", "a") as f:
            f.write("# Verification and Ablation\n\n")
            f.write("This run uses synthetic data generated by `--demo`.\n")
            f.write("Periods: 100 days synthetic. 50/50 Train/Test split for ablation.\n\n")

            f.write("## Ablation Ladder\n")
            f.write(ablation_df.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n")

            f.write("## Drift Flags\n")
            if drift_flags:
                f.write(pd.DataFrame(drift_flags).to_markdown(index=False, floatfmt=".3f"))
            else:
                f.write("No model drift detected.\n")
            f.write("\n\n")

            f.write(replay_report)
            f.write("\n")

        # Create JSON for Dashboard (Amendment 5)
        import datetime
        import json
        import subprocess

        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit_hash = "unknown"

        # Parse the ablation_df to get RMSE drop
        try:
            raw_rmse = float(
                ablation_df.loc[ablation_df["Stage"] == "Raw Best", "Precip RMSE"].values[0]
            )
            full_rmse = float(
                ablation_df.loc[ablation_df["Stage"] == "Full (EMOS)", "Precip RMSE"].values[0]
            )
            blend_rmse_change = ((raw_rmse - full_rmse) / raw_rmse) * 100
        except Exception:
            blend_rmse_change = 0.0

        res_json = {
            "mode": "DEMO",
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "git_commit": commit_hash,
                "data_resolution": float(models.lat[1] - models.lat[0]) if len(models.lat) > 1 else 1.0,
                "data_mode": "synthetic"
            },
            "blend_rmse_change_pct": blend_rmse_change,
            "ablation": ablation_df.to_dict(orient="records"),
            "drift_flags": drift_flags,
        }

        with open("docs/results_demo.json", "w") as f:
            json.dump(res_json, f, indent=2)

        logger.info(
            "Ablation complete. Results appended to docs/results_demo.md and docs/results_demo.json"
        )
    except ImportError:
        logger.warning("Ablation module not yet implemented.")


if __name__ == "__main__":
    main()
