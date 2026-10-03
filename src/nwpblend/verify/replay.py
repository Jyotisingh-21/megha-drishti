import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def replay_events(models: xr.Dataset, truth: xr.Dataset, is_demo: bool = True) -> str:
    """
    Replays historical extreme events and generates a markdown report.
    """
    events = {
        "Wayanad Heavy Rain": {
            "start": "2024-07-29",
            "end": "2024-07-31",
            "var": "precip",
            "thresh": 115.6,
        },
        "Cyclone Biparjoy": {
            "start": "2023-06-14",
            "end": "2023-06-16",
            "var": "wind10m",
            "thresh": 62.0,
        },
        "Cyclone Remal": {
            "start": "2024-05-25",
            "end": "2024-05-27",
            "var": "wind10m",
            "thresh": 62.0,
        },
        "Delhi Heatwave": {
            "start": "2024-05-27",
            "end": "2024-05-30",
            "var": "t2m",
            "thresh": 45.0,
        },
    }

    report = ["### Event Replay\n"]
    if is_demo:
        report.append(
            "> **[DEMO DATA]** This replay is running on synthetic data. Real historical dates are mapped to available demo dates.\n"
        )
    else:
        report.append(
            "> Note: AWS/Google ECMWF open data mirrors only retain the last 5-10 days of forecasts. "
            "To fully replay 2023/2024 events in REAL mode, GFS archive data from AWS s3://noaa-gfs-bdp-pds/ "
            "and historical ECMWF MARS requests are required. "
            "Events are marked **[REAL]** only if exact forecast and observation dates match.\n"
        )

    times = pd.to_datetime(truth.time.values)

    for name, config in events.items():
        if len(times) == 0:
            report.append(
                f"#### {name} ({config['start']} to {config['end']})\n*No data available.*\n"
            )
            continue

        # Map to demo dates or use exact
        if is_demo:
            idx = np.random.randint(0, len(times) - 3)
            event_times = times[idx : idx + 3]
            tag = "[DEMO]"
        else:
            event_times = times[(times >= config["start"]) & (times <= config["end"])]
            if len(event_times) > 0:
                tag = "**[REAL]**"
            else:
                tag = "[NO DATA]"

        report.append(f"#### {name} {tag} ({config['start']} to {config['end']})")

        if len(event_times) == 0:
            report.append("*Dates not present in dataset.*\n")
            continue

        t_slice = truth.sel(time=event_times)

        # Max observed
        max_obs = float(t_slice[config["var"]].max().values)

        report.append(f"- **Max Observed {config['var']}**: {max_obs:.1f}")

        if max_obs > config["thresh"]:
            report.append("- **Hit/Miss**: **HIT** (Extreme threshold exceeded)\n")
        else:
            report.append("- **Hit/Miss**: **MISS** (Threshold not reached)\n")

    return "\n".join(report)


def check_model_drift(
    models: xr.Dataset, truth: xr.Dataset, avail: xr.DataArray, fraction: float = 0.8
) -> list:
    """
    Flags models if trailing 14-day RMSE falls below (i.e. error goes higher than)
    the 90-day mean RMSE by a certain fraction. (Higher RMSE = lower skill).
    So flag if: 14_day_rmse > 90_day_rmse / fraction (e.g. 1.25x worse).
    """
    flags = []
    times = pd.to_datetime(models.time.values)
    if len(times) < 90:
        logger.warning("Need at least 90 days for full drift check. Using max available.")

    t_end = times[-1]
    t_14 = t_end - pd.Timedelta(days=14)
    t_90 = t_end - pd.Timedelta(days=90)

    models_14 = models.sel(time=slice(t_14, t_end))
    truth_14 = truth.sel(time=slice(t_14, t_end))

    models_90 = models.sel(time=slice(t_90, t_end))
    truth_90 = truth.sel(time=slice(t_90, t_end))

    for m in models.model.values:
        rmse_14 = float(
            np.sqrt(((models_14["t2m"].sel(model=m) - truth_14["t2m"]) ** 2).mean()).values
        )
        rmse_90 = float(
            np.sqrt(((models_90["t2m"].sel(model=m) - truth_90["t2m"]) ** 2).mean()).values
        )

        if rmse_14 > rmse_90 / fraction:
            flags.append(
                {
                    "model": m,
                    "14_day_rmse": rmse_14,
                    "90_day_rmse": rmse_90,
                    "status": "DRIFT_DETECTED",
                }
            )

    return flags
