import argparse
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from nwpblend.blend.baselines import bma, equal_weight, ewa
from nwpblend.harmonise.store import stack_models, write_store
from nwpblend.pipeline import run_daily
from nwpblend.truth import fetch_real_truth
from nwpblend.verify.metrics import bias, mae, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    out_lines = []
    
    if args.demo:
        logger.info("Loading demo data for benchmark...")
        try:
            truth = xr.open_zarr("data/demo/truth.zarr").load()

            if not os.path.exists("data/processed/stacked_models.zarr"):
                logger.info("Stacked models not found. Stacking demo data now...")
                expected = ["ecmwf_ifs", "gfs", "aifs", "ncum_g", "graphcast", "pangu"]
                m_dict = {}
                for m in expected:
                    try:
                        m_dict[m] = xr.open_zarr(f"data/demo/models/{m}.zarr").load()
                    except Exception:
                        pass

                if m_dict:
                    stacked = stack_models(m_dict, expected)
                    os.makedirs("data/processed", exist_ok=True)
                    write_store(stacked, "data/processed/stacked_models.zarr")

            models = xr.open_zarr("data/processed/stacked_models.zarr").load()
            times = models.time.values
            if len(times) < 30:
                logger.error("Dataset too small for 30-day test window.")
                return

            test_times = times[-30:]
            truth_test = truth.sel(time=test_times)
            is_demo = True
            
        except Exception as e:
            logger.error(f"Failed to load stacked models or truth. Ensure Prompt 1 is run. {e}")
            return
            
    else:
        logger.info("Loading REAL stacked models...")
        try:
            models = xr.open_zarr("data/processed/stacked_models.zarr").load()
        except Exception as e:
            logger.error(f"Failed to load stacked models. Run fetch_history first. {e}")
            return
            
        times = models.time.values
        start_date = str(pd.to_datetime(times[0]).date())
        end_date = str(pd.to_datetime(times[-1]).date())
        
        logger.info(f"Models span {start_date} to {end_date}. Fetching real truth...")
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)
        domain = cfg["domain"] 
        
        truth = fetch_real_truth(start_date, end_date, domain)
        if truth is None:
            logger.error("Failed to fetch real truth.")
            return
            
        truth_times = truth.time.values
        # Strip timezone from model times if any
        model_times = pd.to_datetime(times).tz_localize(None).values
        intersect_times = np.intersect1d(model_times, truth_times)
        
        if len(intersect_times) == 0:
            logger.error("No overlap between model history and available truth.")
            return
            
        logger.info(f"Scoring {len(intersect_times)} overlapping days.")
        test_times = intersect_times
        truth_test = truth.sel(time=test_times)
        
        # Align models to the same times
        models["time"] = model_times
        models = models.sel(time=test_times).load()
        is_demo = False

    # Baselines
    models_test = models.sel(time=test_times)
    available_test = models.available.sel(time=test_times)

    logger.info("Computing equal weights...")
    eq_blend = equal_weight(models_test, available_test)

    logger.info("Computing EWA...")
    ewa_blend = ewa(models, truth, models.available, window=30).sel(time=test_times)

    logger.info("Computing BMA...")
    bma_blend = bma(models, truth, models.available, window=30).sel(time=test_times)

    mode_str = "DEMO Data" if is_demo else "REAL Data"
    print(f"=== Benchmark Results ({mode_str}) ===")

    out_lines = ["| Model | Var | RMSE | MAE | Bias |", "|-------|-----|------|-----|------|"]

    def evaluate_dataset(name, ds, var, lead):
        if var not in truth_test:
            return
        fcst_val = ds[var].sel(lead=lead)
        truth_val = truth_test[var]

        if fcst_val.shape != truth_val.shape:
            return

        r = float(rmse(fcst_val, truth_val).mean().values)
        m = float(mae(fcst_val, truth_val).mean().values)
        b = float(bias(fcst_val, truth_val).mean().values)

        out_lines.append(f"| {name} | {var} | {r:.3f} | {m:.3f} | {b:.3f} |")

    lead = models.lead.values[0]
    variables = ["t2m", "precip", "wind10m", "gust10m"]

    for var in variables:
        for m_name in models.model.values:
            m_ds = models_test.sel(model=m_name)
            evaluate_dataset(f"Single: {m_name}", m_ds, var, lead)

        evaluate_dataset("Blend: Equal", eq_blend, var, lead)
        evaluate_dataset("Blend: EWA", ewa_blend, var, lead)
        evaluate_dataset("Blend: BMA", bma_blend, var, lead)

    for line in out_lines:
        print(line)

    results_file = "docs/results_demo.md" if is_demo else "docs/results.md"
    json_file = "docs/results_demo.json" if is_demo else "docs/results.json"

    with open(results_file, "w") as f:
        f.write(f"# Benchmark Results ({mode_str})\n\n")
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

        replay_report = replay_events(models, truth, is_demo=is_demo)
        drift_flags = check_model_drift(models, truth, models.available)

        with open(results_file, "a") as f:
            f.write("# Verification and Ablation\n\n")
            if is_demo:
                f.write("This run uses synthetic data generated by `--demo`.\n")
                f.write("Periods: 100 days synthetic. 50/50 Train/Test split for ablation.\n\n")
            else:
                f.write(f"This run uses real data spanning {start_date} to {end_date}.\n")

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

        # Create JSON for Dashboard
        import datetime
        import json
        import subprocess

        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit_hash = "unknown"

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
            "mode": "DEMO" if is_demo else "REAL",
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "git_commit": commit_hash,
                "data_resolution": float(models.lat[1] - models.lat[0])
                if len(models.lat) > 1
                else 1.0,
                "data_mode": "synthetic" if is_demo else "real",
                "start_date": str(start_date) if not is_demo else None,
                "end_date": str(end_date) if not is_demo else None,
                "scored_days": len(test_times)
            },
            "blend_rmse_change_pct": blend_rmse_change,
            "ablation": ablation_df.to_dict(orient="records"),
            "drift_flags": drift_flags,
        }

        with open(json_file, "w") as f:
            json.dump(res_json, f, indent=2)

        logger.info(f"Ablation complete. Results appended to {results_file} and {json_file}")
    except ImportError:
        logger.warning("Ablation module not yet implemented.")


if __name__ == "__main__":
    main()
