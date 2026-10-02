import glob
import json
import os

import streamlit as st
from data_loader import load_data

st.set_page_config(layout="wide", page_title="Megha-Drishti | Command Center")

# --- DATA LOADING ---
def get_metrics():
    metrics = {
        "models_online": 0,
        "rmse_change": "N/A",
        "last_run": "Never",
        "is_demo": False
    }
    
    # Get last run report
    reports = sorted(glob.glob("data/logs/run_report_*.json"))
    if reports:
        with open(reports[-1], "r") as f:
            last_run = json.load(f)
            metrics["last_run"] = last_run.get("date", "Unknown")
            metrics["is_demo"] = (last_run.get("mode") == "DEMO")
            # For the demo, we assume the 6 expected models are online if harmonise succeeded
            if "harmonise" in last_run.get("stages", {}):
                metrics["models_online"] = 6 if last_run["stages"]["harmonise"]["status"] == "SUCCESS" else 0

    # Get RMSE improvement
    res_file = "docs/results_demo.json" if metrics["is_demo"] else "docs/results.json"
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            res = json.load(f)
            chg = res.get("blend_rmse_change_pct", 0)
            metrics["rmse_change"] = f"{chg:.1f}%"
            
    return metrics

metrics = get_metrics()

# --- HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Command Center")
    if metrics["is_demo"]:
        st.caption("🔴 RUNNING IN DEMO MODE")
with col2:
    st.metric("Last Run", metrics["last_run"])

# --- KPIS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Models Online", metrics["models_online"], "+0")
with kpi2:
    st.metric("Today's Regime", "Monsoon (Active)", "Stable")
with kpi3:
    st.metric("Active Alerts", "12", "+3")
with kpi4:
    st.metric("Blend RMSE vs Best", metrics["rmse_change"], "Improved", delta_color="inverse")

st.divider()

col_map, col_risk = st.columns([2, 1])

with col_map:
    st.subheader("National Alert Thumbnail")
    st.info("National hazard aggregation map goes here (using config thresholds).")

with col_risk:
    st.subheader("Top 5 Regions at Risk")
    st.dataframe(
        {
            "Region (State Box)": ["Western Ghats", "Assam", "Delhi", "Odisha Coast", "Rajasthan"],
            "Hazard": ["Heavy Rain", "Heavy Rain", "Heatwave", "High Wind", "Heatwave"],
            "Probability": ["94%", "88%", "82%", "76%", "71%"]
        },
        hide_index=True
    )
