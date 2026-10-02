import os

import streamlit as st
from components.styles import apply_global_styles
from components.ui import render_header
from data_loader import load_data

st.set_page_config(page_title="Megha-Drishti", page_icon="☁️", layout="wide")

apply_global_styles()

ds, is_demo = load_data()
render_header(is_demo)

if "language" not in st.session_state:
    st.session_state["language"] = "English"

# Render sidebar language toggle
with st.sidebar:
    st.session_state["language"] = st.radio("Language / भाषा", ["English", "Hindi"], index=0 if st.session_state["language"] == "English" else 1)
    if st.session_state["language"] == "Hindi":
        st.warning("Note: Hindi translations need native-speaker review before operational use.")

# Navigation
pages = {
    "Operations": [
        st.Page("pages/00_command_center.py", title="Command Center", icon="🏠"),
        st.Page("pages/01_forecast_map.py", title="Forecast Map", icon="🗺️"),
        st.Page("pages/05_point_explorer.py", title="Point Explorer", icon="🔍"),
        st.Page("pages/03_extreme_guidance.py", title="Extreme Guidance", icon="🚨"),
    ],
    "Analysis": [
        st.Page("pages/02_weight_maps.py", title="Weight & Explainability", icon="🧠"),
        st.Page("pages/04_event_replay.py", title="Event Replay", icon="⏪"),
        st.Page("pages/06_skill_drift.py", title="Skill & Drift", icon="📈"),
        st.Page("pages/07_pipeline_status.py", title="Pipeline Status", icon="⚙️"),
        st.Page("pages/06_about.py", title="About", icon="ℹ️"),
    ],
}

pg = st.navigation(pages)
pg.run()
