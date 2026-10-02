import streamlit as st
import plotly.graph_objects as go
from data_loader import load_data
from geo import get_india_borders_trace, get_cities_trace

st.title("Forecast Map")

ds, is_demo = load_data()
var = st.selectbox("Variable", list(ds.data_vars.keys()) if ds.data_vars else ["t2m"])

leads = ds.lead.values if "lead" in ds.dims else [0]
# format leads beautifully
lead_idx = st.select_slider("Lead Time", options=range(len(leads)), format_func=lambda x: str(leads[x]))
lead = leads[lead_idx]

sub = ds[var].isel(time=0)
if "lead" in sub.dims:
    sub = sub.sel(lead=lead)
if "model" in sub.dims:
    sub = sub.mean(dim="model")

# Variable-specific color scales
colorscale = "Viridis"
if var == "precip":
    colorscale = [[0.0, "white"], [0.2, "#a0c4ff"], [0.5, "#4361ee"], [1.0, "#3a0ca3"]]
elif var == "t2m":
    colorscale = [[0.0, "#03045e"], [0.3, "#0077b6"], [0.5, "#00b4d8"], [0.7, "#fdf0d5"], [1.0, "#d00000"]]
elif "wind" in var:
    colorscale = "Teal"

fig = go.Figure()

# Background ocean/map
fig.update_layout(
    plot_bgcolor="#0A192F",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
)

fig.add_trace(go.Contour(
    z=sub.values,
    x=sub.lon.values,
    y=sub.lat.values,
    colorscale=colorscale,
    opacity=0.8,
    line_smoothing=0.85,
    contours=dict(showlines=False),
    colorbar=dict(title=sub.attrs.get('units', ''), x=1.0)
))

fig.add_trace(get_india_borders_trace()); fig.add_trace(get_cities_trace())

fig.update_layout(title=f"Blended {var.upper()}")

st.plotly_chart(fig, use_container_width=True)
