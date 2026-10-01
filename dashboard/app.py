import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr
from i18n import get_text

st.set_page_config(page_title="Megha-Drishti", layout="wide")


# ==========================================
# Data Loading (Direct from output dir)
# ==========================================
@st.cache_data
def load_data():
    is_demo = True
    ds = None
    if os.path.exists("../data/processed/stacked_models.zarr"):
        try:
            ds = xr.open_zarr("../data/processed/stacked_models.zarr").load()
            is_demo = True  # Forcing demo flag for project scope since we used synthetic
        except Exception:
            pass

    if ds is None:
        try:
            # Fallback to a single model if stack doesn't exist
            ds = xr.open_zarr("../data/demo/models/ecmwf_ifs.zarr").load()
        except Exception:
            # Create extreme dummy mock for layout testing
            ds = xr.Dataset(
                {"t2m": (["time", "lead", "lat", "lon"], np.random.normal(30, 2, (1, 3, 20, 20)))},
                coords={
                    "time": [pd.Timestamp.now()],
                    "lead": [0, 24, 48],
                    "lat": np.linspace(8, 38, 20),
                    "lon": np.linspace(68, 98, 20),
                },
            )
    return ds, is_demo


ds, is_demo = load_data()

# ==========================================
# Sidebar & Header
# ==========================================
st.sidebar.title("Megha-Drishti")
if is_demo:
    st.sidebar.warning("⚠️ **DEMO MODE**\n\nOperating on offline synthetic data.")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Forecast",
        "2. Weight Maps",
        "3. Extreme Guidance",
        "4. Skill and Drift",
        "5. Explainability",
        "6. About",
    ],
)

# ==========================================
# 1. Forecast Page
# ==========================================
if page == "1. Forecast":
    st.title("Blended Forecast")

    var = st.selectbox("Variable", list(ds.data_vars.keys()) if ds.data_vars else ["t2m"])
    leads = ds.lead.values if "lead" in ds.dims else [0]
    lead = st.select_slider("Lead Time (h)", options=leads)

    # Map
    sub = ds[var].isel(time=0)
    if "lead" in sub.dims:
        sub = sub.sel(lead=lead)
    if "model" in sub.dims:
        # Mock blend by mean
        sub = sub.mean(dim="model")

    df = sub.to_dataframe().reset_index()
    fig = px.imshow(
        sub.values,
        x=sub.lon.values,
        y=sub.lat.values,
        color_continuous_scale="Viridis",
        origin="lower",
        title=f"Blended {var.upper()} at Lead {lead}h",
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 2. Weight Maps
# ==========================================
elif page == "2. Weight Maps":
    st.title("Dynamic Model Weights")
    st.write("Displays the spatial trust (weights) assigned to each model by the gating network.")

    models = ds.model.values if "model" in ds.dims else ["ecmwf_ifs"]
    selected_model = st.selectbox("View weights for:", models)

    # Generate mock weights that sum to 1
    # In reality, this would read the `weights` DataArray
    lats = ds.lat.values
    lons = ds.lon.values
    mock_weights = np.random.dirichlet(np.ones(len(models)), size=(len(lats), len(lons)))

    model_idx = list(models).index(selected_model) if selected_model in models else 0
    w_map = mock_weights[:, :, model_idx]

    fig = px.imshow(
        w_map,
        x=lons,
        y=lats,
        color_continuous_scale="Blues",
        origin="lower",
        title=f"Weight Map: {selected_model}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Current Regime: **Active Monsoon** (Derived from PCA+KMeans proxy)")

# ==========================================
# 3. Extreme Guidance
# ==========================================
elif page == "3. Extreme Guidance":
    st.title("Extreme Event Probabilities")

    lang = st.radio("Language / भाषा", ["en", "hi"], horizontal=True)
    event_type = st.selectbox(
        "Event",
        ["heavy_rain_alert", "heatwave_alert", "high_wind_alert"],
        format_func=lambda x: get_text(lang, x),
    )

    st.subheader(f"Probability Map: {get_text(lang, event_type)}")

    # Mock probability map
    lats, lons = ds.lat.values, ds.lon.values
    prob = np.random.uniform(0, 1, size=(len(lats), len(lons)))
    fig = px.imshow(
        prob, x=lons, y=lats, color_continuous_scale="Reds", origin="lower", zmin=0, zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-Risk Districts")
    dist_df = pd.DataFrame(
        {
            "District": ["Wayanad", "Mumbai", "Delhi", "Chennai"],
            "Probability": [0.85, 0.65, 0.10, 0.05],
            "Warning Level": [
                get_text(lang, "level_red"),
                get_text(lang, "level_orange"),
                get_text(lang, "level_green"),
                get_text(lang, "level_green"),
            ],
        }
    )
    st.dataframe(dist_df, use_container_width=True)

# ==========================================
# 4. Skill and Drift
# ==========================================
elif page == "4. Skill and Drift":
    st.title("Model Skill and Drift")

    try:
        with open("../docs/results_demo.md", "r") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("docs/results_demo.md not found. Run `scripts/run_benchmark.py` first.")

# ==========================================
# 5. Explainability
# ==========================================
elif page == "5. Explainability":
    st.title("SHAP Feature Importance")
    st.write("Explains the decisions of the spatial gating network for yesterday's forecast.")

    # Mock SHAP bar chart
    features = ["Spatial MAE (Trailing)", "Temp Anomaly", "Wind Speed", "Elevation", "Time of Year"]
    importance = [0.45, 0.25, 0.15, 0.10, 0.05]

    df = pd.DataFrame({"Feature": features, "Mean |SHAP|": importance})
    fig = px.bar(
        df, x="Mean |SHAP|", y="Feature", orientation="h", title="Global Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. About
# ==========================================
elif page == "6. About":
    st.title("About Megha-Drishti")
    if is_demo:
        st.error(
            "**DEMO MODE ACTIVE**: You are viewing synthetically generated data, not real meteorological forecasts."
        )

    st.markdown("""
    **Megha-Drishti** is a Hybrid AI-NWP Forecast Blending System developed for MoES-NCMRWF.
    
    ### Data Sources
    - **Physical NWP**: ECMWF IFS, NOAA GFS, NCUM-G.
    - **AI Models**: ECMWF AIFS, GraphCast, Pangu-Weather.
    
    ### Pipeline Architecture
    1. **Harmonization**: 0.25° grid normalisation via xESMF.
    2. **Bias Correction**: Quantile mapping & Lapse rate corrections.
    3. **Regimes**: Synoptic clustering (PCA+KMeans).
    4. **Blending**: Spatial Mixture-of-Experts neural network.
    5. **Calibration**: EMOS scaling for probabilistic bounds.
    """)
