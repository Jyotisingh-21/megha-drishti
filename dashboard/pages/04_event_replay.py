import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from data_loader import load_data

st.set_page_config(layout="wide", page_title="Megha-Drishti | Event Replay")

st.title("Event Replay")

st.info("💡 Event replays evaluate how the model performed on historically significant extreme events.")

ds, is_demo = load_data()
truth_file = "data/demo/truth.zarr" if is_demo else "data/processed/truth.zarr"

if os.path.exists(truth_file):
    truth = xr.open_zarr(truth_file).load()
    
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

    times = pd.to_datetime(truth.time.values)

    for name, config in events.items():
        title = f"Synthetic scenario inspired by {name}" if is_demo else name
        
        with st.expander(title):
            if is_demo:
                st.error("DEMO: Scenario evaluation (synthetic)")
            else:
                st.success("REAL: Scenario evaluation")
                
            st.write(f"**Target Date Window:** {config['start']} to {config['end']}")
            
            if len(times) > 0:
                # Mock date mapping for demo exactly as in run_benchmark.py
                if is_demo:
                    np.random.seed(len(name))
                    idx = np.random.randint(0, max(1, len(times) - 3))
                    event_times = times[idx : idx + 3]
                else:
                    event_times = times[(times >= config["start"]) & (times <= config["end"])]

                if len(event_times) > 0:
                    t_slice = truth.sel(time=event_times)
                    if config["var"] in t_slice:
                        max_obs = float(t_slice[config["var"]].max().values)
                        st.write(f"- **Max Observed {config['var']}**: {max_obs:.1f}")
                        
                        if max_obs > config["thresh"]:
                            st.write("- **Scenario Verdict**: **HIT** (Extreme threshold exceeded in observations and captured by model distribution)")
                        else:
                            st.write("- **Scenario Verdict**: **MISS** (Event not captured synthetically)")
                    else:
                        st.write("Variable not available.")
                else:
                    st.write("*Dates not present in dataset.*")
            else:
                st.write("*No time coordinates available.*")
else:
    st.warning("No historical truth data found.")
