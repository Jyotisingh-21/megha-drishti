import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="Megha-Drishti | Skill & Drift")

st.title("Skill and Drift")

# Use metrics from docs/results_demo.json
data = []
if os.path.exists("docs/results_demo.json"):
    with open("docs/results_demo.json", "r") as f:
        res = json.load(f)
        for row in res.get("ablation_ladder", []):
            data.append(
                {
                    "Stage": row.get("Step", row.get("Stage")),
                    "RMSE": row.get("RMSE", row.get("Precip RMSE")),
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
# Mock drift line chart
dates = pd.date_range("2024-01-01", periods=30)
df_drift = pd.DataFrame(
    {
        "Date": dates,
        "GFS Bias": np.random.normal(0, 1, 30).cumsum(),
        "ECMWF Bias": np.random.normal(0, 0.5, 30).cumsum(),
    }
)
fig_d = px.line(df_drift, x="Date", y=["GFS Bias", "ECMWF Bias"], title="30-Day Rolling Bias")
fig_d.update_layout(plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_d, use_container_width=True)
