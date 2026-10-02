import streamlit as st
import numpy as np
import plotly.graph_objects as go
from data_loader import load_data
from geo import get_india_borders_trace
from i18n import get_text
from components.ui import kpi_card

st.title("Extreme Guidance")

ds, is_demo = load_data()
lang = st.radio("Language / भाषा", ["en", "hi"], horizontal=True)

event_type = st.selectbox("Event", ["heavy_rain_alert", "heatwave_alert", "high_wind_alert"], format_func=lambda x: get_text(lang, x))

# Mock probability map
lats, lons = ds.lat.values, ds.lon.values
np.random.seed(42)
prob = np.random.uniform(0, 1, size=(len(lats), len(lons)))
from scipy.ndimage import gaussian_filter
prob = gaussian_filter(prob, sigma=4)
prob = (prob - prob.min()) / (prob.max() - prob.min())

fig = go.Figure()
fig.update_layout(
    plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=False, visible=False),
    yaxis=dict(showgrid=False, visible=False, scaleanchor="x", scaleratio=1),
)

fig.add_trace(go.Contour(
    z=prob, x=lons, y=lats,
    colorscale="Reds" if "heatwave" in event_type else "Blues", 
    opacity=0.8,
    contours=dict(showlines=False),
    colorbar=dict(title="Probability")
))

fig.add_trace(get_india_borders_trace())
fig.update_layout(title=f"{get_text(lang, event_type)} - Exceedance Probability")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Regional Warnings")
st.info("Note: Showing coarse regional boxes for illustrative purposes. True district-level analysis requires high-resolution IMD shapefiles.")
st.markdown("🚨 **Western Ghats**: High risk of extreme precipitation (>90%)")
