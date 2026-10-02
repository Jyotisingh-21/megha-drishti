import streamlit as st
import os

from components.styles import apply_global_styles
from components.ui import render_header
from data_loader import load_data

st.set_page_config(page_title="Megha-Drishti", page_icon="☁️", layout="wide")

apply_global_styles()

ds, is_demo = load_data()
render_header(is_demo)

# Navigation
pages = {
    "Dashboard": [
        st.Page("pages/01_forecast_map.py", title="Forecast Map", icon="🗺️"),
        st.Page("pages/02_weight_maps.py", title="Weight Maps", icon="⚖️"),
        st.Page("pages/03_extreme_guidance.py", title="Extreme Guidance", icon="⚠️"),
    ],
    "Analysis": [
        st.Page("pages/04_skill_drift.py", title="Skill & Drift", icon="📈"),
        st.Page("pages/05_explainability.py", title="Explainability", icon="🧠"),
        st.Page("pages/06_about.py", title="About", icon="ℹ️"),
    ]
}

pg = st.navigation(pages)
pg.run()
