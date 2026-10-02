import glob
import json
import os
import yaml
import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from data_loader import load_data

st.set_page_config(layout="wide", page_title="Megha-Drishti | Command Center")

ds, is_demo = load_data()

# --- DATA LOADING ---
def get_metrics(ds, is_demo):
    metrics = {"models_online": 0, "rmse_change": "N/A", "last_run": "Never", "is_demo": is_demo}

    # Get last run report
    reports = sorted(glob.glob("data/logs/run_report_*.json"))
    if reports:
        with open(reports[-1], "r") as f:
            last_run = json.load(f)
            metrics["last_run"] = last_run.get("date", "Unknown")
            # For the demo, we assume the 6 expected models are online if harmonise succeeded
            if "harmonise" in last_run.get("stages", {}):
                metrics["models_online"] = (
                    6 if last_run["stages"]["harmonise"]["status"] == "SUCCESS" else 0
                )
    else:
        metrics["models_online"] = len(ds.model.values) if "model" in ds.dims else 1

    # Get RMSE improvement
    res_file = "docs/results_demo.json" if metrics["is_demo"] else "docs/results.json"
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            res = json.load(f)
            chg = res.get("blend_rmse_change_pct", 0)
            metrics["rmse_change"] = f"{chg:.1f}%"

    # Get Today's Regime
    regime_file = "data/demo/regimes.zarr" if is_demo else "data/processed/regimes.zarr"
    if os.path.exists(regime_file):
        reg = xr.open_zarr(regime_file)
        if len(reg.time.values) > 0:
            reg_val = int(reg.cluster.isel(time=-1).values)
            mapping = {0: "Pre-Monsoon", 1: "Monsoon", 2: "Post-Monsoon", 3: "Winter"}
            metrics["regime"] = mapping.get(reg_val, f"Cluster {reg_val}")
    else:
        metrics["regime"] = "Unknown"
        
    return metrics

metrics = get_metrics(ds, is_demo)

# Calculate dynamic active alerts and regions at risk
with open("configs/thresholds.yaml", "r") as f:
    thresholds = yaml.safe_load(f)

# Use Day 1 (lead=24h/1d) or 0
sub = ds.isel(time=0)
if "lead" in sub.dims:
    sub = sub.isel(lead=0)

if "model" in sub.dims:
    sub = sub.mean(dim="model") # simple ensemble mean for alerts

precip_thr = thresholds["precip_mm_24h"]["heavy"]
t2m_thr = thresholds["heatwave_celsius"]["absolute"]["plains"]

# Count points exceeding thresholds
try:
    rain_alerts = (sub["precip"] > precip_thr).sum().values
    heat_alerts = (sub["t2m"] > t2m_thr).sum().values
except Exception:
    rain_alerts = 0
    heat_alerts = 0

active_alerts = int(rain_alerts + heat_alerts)

# Generate risk regions list (by flattening the grid)
try:
    p_flat = sub["precip"].values.flatten()
    t_flat = sub["t2m"].values.flatten()
    lats = np.broadcast_to(sub.lat.values[:, None], (len(sub.lat), len(sub.lon))).flatten()
    lons = np.broadcast_to(sub.lon.values[None, :], (len(sub.lat), len(sub.lon))).flatten()
    
    risk_list = []
    # top 3 rain
    p_idx = np.argsort(p_flat)[-3:][::-1]
    for i in p_idx:
        if p_flat[i] > precip_thr:
            risk_list.append({"Location": f"{lats[i]:.1f}°N, {lons[i]:.1f}°E", "Hazard": "Heavy Rain", "Value": f"{p_flat[i]:.1f} mm"})
    # top 2 heat
    t_idx = np.argsort(t_flat)[-2:][::-1]
    for i in t_idx:
        if t_flat[i] > t2m_thr:
            risk_list.append({"Location": f"{lats[i]:.1f}°N, {lons[i]:.1f}°E", "Hazard": "Heatwave", "Value": f"{t_flat[i]:.1f} °C"})
except Exception:
    risk_list = []

# --- HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Command Center")
    if metrics["is_demo"]:
        st.caption("🚀 RUNNING IN DEMO MODE")
with col2:
    st.metric("Last Run", metrics["last_run"])

# --- KPIS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Models Online", metrics["models_online"])
with kpi2:
    st.metric("Today's Regime", metrics["regime"])
with kpi3:
    st.metric("Active Alerts (Grid cells)", active_alerts)
with kpi4:
    st.metric("Blend RMSE vs Best", metrics["rmse_change"], delta_color="inverse")

st.divider()

col_map, col_risk = st.columns([2, 1])

with col_map:
    st.subheader("National Alert Thumbnail")
    if active_alerts > 0:
        fig = px.imshow((sub["precip"] > precip_thr).values, color_continuous_scale="Reds")
        fig.update_layout(title="Heavy Rain Alerts", coloraxis_showscale=False, plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active hazards on map.")

with col_risk:
    st.subheader("Top Regions at Risk")
    if risk_list:
        st.dataframe(risk_list, hide_index=True, use_container_width=True)
    else:
        st.success("No extreme points exceeding thresholds.")
