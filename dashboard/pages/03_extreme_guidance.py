import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
import yaml
from data_loader import load_data
from geo import get_india_borders_trace

st.set_page_config(layout="wide", page_title="Megha-Drishti | Extreme Guidance")

st.title("Extreme Guidance")

with open("configs/thresholds.yaml", "r") as f:
    thresholds = yaml.safe_load(f)

hazard = st.radio(
    "Hazard", ["Heavy Rain (>64.5mm)", "Heatwave (>40°C)", "Strong Wind (>30km/h)"], horizontal=True
)

if "Rain" in hazard:
    var = "precip"
    thresh = thresholds["precip_mm_24h"]["heavy"]
elif "Heat" in hazard:
    var = "t2m"
    thresh = thresholds["heatwave_celsius"]["absolute"]["plains"]
else:
    var = "wind10m"
    thresh = thresholds["wind_gusts_kmh"]["strong"] * (1000 / 3600)  # km/h to m/s

ds, is_demo = load_data()
if var not in ds.data_vars:
    var = next(iter(ds.data_vars.keys()))

leads = ds.lead.values if "lead" in ds.dims else [0]
lead_idx = st.select_slider(
    "Lead Time",
    options=range(len(leads)),
    format_func=lambda x: f"Day {int(leads[x])}" if leads[x] > 0 else "Analysis",
)

sub = ds[var].isel(time=0)
if "lead" in sub.dims:
    sub = sub.sel(lead=leads[lead_idx])

if "model" in sub.dims:
    # Probability of exceeding threshold across ensemble
    prob = (sub > thresh).mean(dim="model")
else:
    # Mock probability from single model
    prob = xr.where(sub > thresh, 0.8, 0.1)

st.info(
    f"💡 Showing probability of exceeding {thresh:.1f} for {var} based on configured IMD thresholds."
)

fig = go.Figure()

# Custom colorscale using alert_colors
colors = thresholds["alert_colors"]
cscale = [
    [0.0, "green"],
    [colors["yellow"], "yellow"],
    [colors["orange"], "orange"],
    [colors["red"], "red"],
    [1.0, "darkred"],
]

fig.add_trace(
    go.Contour(
        z=prob.values,
        x=ds.lon.values,
        y=ds.lat.values,
        colorscale=cscale,
        opacity=0.7,
        contours={"showlines": False},
        colorbar={"title": "Probability", "tickformat": ".0%"},
    )
)

borders = get_india_borders_trace()
fig.add_trace(
    go.Scatter(
        x=borders.x,
        y=borders.y,
        mode="lines",
        line=borders.line,
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.update_layout(
    plot_bgcolor="#0A192F",
    paper_bgcolor="rgba(0,0,0,0)",
    margin={"l": 0, "r": 0, "t": 30, "b": 0},
    xaxis={"showgrid": False, "visible": False},
    yaxis={"showgrid": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
)

st.plotly_chart(fig, use_container_width=True)
