import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr
from data_loader import load_data

st.set_page_config(layout="wide", page_title="Megha-Drishti | Skill & Drift")

st.title("Skill and Drift")

# Use metrics from docs/results_demo.json
data = []
if os.path.exists("docs/results_demo.json"):
    with open("docs/results_demo.json", "r") as f:
        res = json.load(f)
        for row in res.get("ablation", []):
            data.append(
                {
                    "Stage": row.get("Step", row.get("Stage")),
                    "RMSE": row.get("RMSE", row.get("Precip RMSE", 0)),
                }
            )

if data:
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Stage", y="RMSE", title="Ablation Ladder: RMSE by Pipeline Stage")
    fig.update_layout(plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No benchmark metrics available. Run scripts/run_benchmark.py --demo first.")

st.divider()

st.subheader("Model Drift Monitoring")
st.write("Tracks systematic errors and biases in individual models over time.")

# Compute real drift instead of mock
ds, is_demo = load_data()

st.sidebar.warning("RUNNING IN DEMO MODE") if is_demo else st.sidebar.success("RUNNING IN REAL MODE")
truth_file = "data/demo/truth.zarr" if is_demo else "data/processed/truth.zarr"

if os.path.exists(truth_file) and "model" in ds.dims:
    truth = xr.open_zarr(truth_file).load()
    var = "t2m"
    if var in ds and var in truth:
        # Calculate daily spatial mean bias per model
        # bias = fcst - truth
        fcst = ds[var].isel(lead=0) if "lead" in ds.dims else ds[var]

        # align lat lons
        fcst_aligned, truth_aligned = xr.align(fcst, truth[var], join="inner")

        bias = (fcst_aligned - truth_aligned).mean(dim=["lat", "lon"])

        # Pandas dataframe
        df_drift = bias.to_dataframe().reset_index()
        # Pivot
        df_pivot = df_drift.pivot(index="time", columns="model", values=var).reset_index()

        # 7-day rolling mean
        df_roll = df_pivot.set_index("time").rolling(window=7, min_periods=1).mean().reset_index()

        fig_d = px.line(
            df_roll, x="time", y=df_roll.columns[1:], title="7-Day Rolling Spatial Mean Bias (T2M)"
        )
        fig_d.update_layout(plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d, use_container_width=True)
else:
    st.info("No time-series data available for drift tracking.")
