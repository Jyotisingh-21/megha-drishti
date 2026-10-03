import argparse
import glob
import json
import logging
import os
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xarray as xr

from nwpblend.blend.baselines import bma, equal_weight, ewa
from nwpblend.verify.metrics import bias, mae, rmse


def bootstrap_metric(metric_func, fcst, truth, n_boot=100):
    vals = []

    # Actually just bootstrap over the time dimension for simplicity
    n_time = fcst.shape[0]
    for _ in range(n_boot):
        idx = np.random.randint(0, n_time, size=n_time)
        f_samp = fcst.isel(time=idx)
        t_samp = truth.isel(time=idx)

        m_val = metric_func(f_samp, t_samp).mean().item()
        vals.append(m_val)

    return np.mean(vals), np.percentile(vals, 5), np.percentile(vals, 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    os.makedirs("data/logs", exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"data/logs/benchmark_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.demo:
        logger.info("Loading demo pairs...")
        try:
            m_dict = {}
            for m in ["ecmwf_ifs", "gfs", "aifs", "ncum_g", "graphcast", "pangu"]:
                if os.path.exists(f"data/demo/models/{m}.zarr"):
                    m_dict[m] = xr.open_zarr(f"data/demo/models/{m}.zarr")

            from nwpblend.harmonise.store import stack_models

            models = stack_models(m_dict, list(m_dict.keys())).load()
            truth = xr.open_zarr("data/demo/truth.zarr").load()
            # Demo sets them up, need to rename truth for the rest of script
            truth = truth.rename({v: f"truth_{v}" for v in truth.data_vars})
            n_pairs = len(models.time)
            start_date = str(pd.to_datetime(models.time.values[0]).date())
            end_date = str(pd.to_datetime(models.time.values[-1]).date())
            truth_source = "Synthetic Demo Data"
        except Exception as e:
            logger.error(f"Failed to load demo data: {e}")
            return
    else:
        logger.info("Loading verified pairs...")

        pair_dirs = sorted(glob.glob("data/verified/pair_*"))
        if not pair_dirs:
            logger.error("No verified pairs found in data/verified/")
            return

        hist_m = []
        hist_t = []

        for pd_dir in pair_dirs:
            m_path = os.path.join(pd_dir, "models.zarr")
            t_path = os.path.join(pd_dir, "truth.zarr")
            if os.path.exists(m_path) and os.path.exists(t_path):
                try:
                    hist_m.append(xr.open_zarr(m_path))
                    hist_t.append(xr.open_zarr(t_path))
                except Exception as e:
                    logger.warning(f"Failed to load {pd_dir}: {e}")

        if not hist_m:
            logger.error("No valid verified pairs could be loaded.")
            return

        models = xr.concat(hist_m, dim="time").load()
        truth = xr.concat(hist_t, dim="time").load()

        n_pairs = len(models.time)
        start_date = str(pd.to_datetime(models.time.values[0]).date())
        end_date = str(pd.to_datetime(models.time.values[-1]).date())
        truth_source = "IMD (precip), ARCO-ERA5 (other vars)"

    logger.info(f"Loaded {n_pairs} pairs from {start_date} to {end_date}.")

    out_lines = [
        f"**Pairs:** {n_pairs} | **Period:** {start_date} to {end_date} | **Truth:** {truth_source}\n",
        "| Model | Var | RMSE (90% CI) | MAE (90% CI) | Bias (90% CI) |",
        "|-------|-----|---------------|--------------|---------------|",
    ]

    json_results = []

    # Evaluate over all available time
    models_test = models
    truth_test = truth
    available_test = models.available

    logger.info("Computing equal weights...")
    eq_blend = equal_weight(models_test, available_test)

    # Rename truth back to normal names for EWA/BMA
    truth_unrenamed = truth.rename({v: v.replace("truth_", "") for v in truth.data_vars})

    logger.info("Computing EWA...")
    ewa_blend = ewa(models, truth_unrenamed, models.available, window=30)

    logger.info("Computing BMA...")
    bma_blend = bma(models, truth_unrenamed, models.available, window=30)

    def evaluate_dataset(name, ds, var, lead):
        truth_var = var if var in truth_test else f"truth_{var}"
        if truth_var not in truth_test:
            return

        fcst_val = ds[var].sel(lead=lead)
        truth_val = truth_test[truth_var]

        if fcst_val.shape != truth_val.shape:
            return

        # Bootstrap
        r_m, r_l, r_h = bootstrap_metric(rmse, fcst_val, truth_val, 20)
        m_m, m_l, m_h = bootstrap_metric(mae, fcst_val, truth_val, 20)
        b_m, b_l, b_h = bootstrap_metric(bias, fcst_val, truth_val, 20)

        out_lines.append(
            f"| {name} | {var} | {r_m:.2f} ({r_l:.2f}-{r_h:.2f}) | {m_m:.2f} ({m_l:.2f}-{m_h:.2f}) | {b_m:.2f} ({b_l:.2f}-{b_h:.2f}) |"
        )

        json_results.append(
            {
                "model": name,
                "variable": var,
                "rmse": r_m,
                "rmse_ci": [r_l, r_h],
                "mae": m_m,
                "mae_ci": [m_l, m_h],
                "bias": b_m,
                "bias_ci": [b_l, b_h],
            }
        )

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

    results_file = "docs/results.md"
    json_file = "docs/results.json"

    with open(results_file, "w") as f:
        f.write("# Benchmark Results (REAL Data)\n\n")
        f.write("\n".join(out_lines))
        f.write("\n")

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_hash = "unknown"

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": commit_hash,
        "mode": "REAL",
        "n_pairs": n_pairs,
        "period": [start_date, end_date],
        "metrics": json_results,
    }
    with open(json_file, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
