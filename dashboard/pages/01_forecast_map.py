import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data
from geo import get_cities_trace, get_india_borders_trace
from plotly.subplots import make_subplots

st.title("Forecast Map")

ds, is_demo = load_data()
variables = [v for v in ds.data_vars.keys() if v != "available"] if ds.data_vars else ["t2m"]

var = st.radio("Variable", variables, horizontal=True)

compare_mode = st.checkbox("Side-by-Side Compare Mode")

if not compare_mode:
    layer = st.radio("Layer", ["Blended", "Individual Models", "Ensemble Spread", "Uncertainty (P90-P10)"], horizontal=True)
else:
    col1, col2 = st.columns(2)
    with col1:
        layer1 = st.selectbox("Left Map", ["Blended", "Ensemble Spread"] + list(ds.model.values) if "model" in ds.dims else [])
    with col2:
        layer2 = st.selectbox("Right Map", ["Blended", "Ensemble Spread"] + list(ds.model.values) if "model" in ds.dims else [])

leads = ds.lead.values if "lead" in ds.dims else [0]
lead_idx = st.select_slider("Lead Time (Scrub to animate)", options=range(len(leads)), format_func=lambda x: f"Day {int(leads[x])}" if leads[x] > 0 else "Analysis")
lead = leads[lead_idx]

def get_layer_data(l_name, dset, variable, l_val):
    s = dset[variable].isel(time=0)
    if "lead" in s.dims:
        s = s.sel(lead=l_val)
    if l_name == "Blended":
        return s.mean(dim="model") if "model" in s.dims else s
    elif l_name == "Ensemble Spread":
        return s.std(dim="model") if "model" in s.dims else s
    elif l_name == "Uncertainty (P90-P10)":
        return s.quantile(0.9, dim="model") - s.quantile(0.1, dim="model") if "model" in s.dims else s
    else:
        return s.sel(model=l_name) if "model" in s.dims else s

colorscale = "Viridis"
if var == "precip":
    colorscale = [[0.0, "white"], [0.2, "#a0c4ff"], [0.5, "#4361ee"], [1.0, "#3a0ca3"]]
elif var == "t2m":
    colorscale = [[0.0, "#03045e"], [0.3, "#0077b6"], [0.5, "#00b4d8"], [0.7, "#fdf0d5"], [1.0, "#d00000"]]
elif "wind" in var:
    colorscale = "Teal"

def add_map_traces(fig, sub, row=1, col=1):
    fig.add_trace(go.Contour(
        z=sub.values, x=ds.lon.values, y=ds.lat.values,
        colorscale=colorscale, opacity=0.8,
        contours={"showlines": False},
        colorbar={"title": sub.attrs.get('units', ''), "x": 1.0 if col==1 and not compare_mode else (1.0 if col==2 else 0.45)}
    ), row=row, col=col)
    
    borders = get_india_borders_trace()
    fig.add_trace(go.Scatter(x=borders.x, y=borders.y, mode="lines", line=borders.line, hoverinfo="skip", showlegend=False), row=row, col=col)
    
    cities = get_cities_trace()
    fig.add_trace(go.Scatter(x=cities.x, y=cities.y, mode="markers+text", marker=cities.marker, text=cities.text, textposition="top center", textfont=cities.textfont, hoverinfo="text", showlegend=False), row=row, col=col)
    
    grid_lons, grid_lats = np.meshgrid(ds.lon.values, ds.lat.values)
    fig.add_trace(go.Scatter(
        x=grid_lons.flatten(),
        y=grid_lats.flatten(),
        mode='markers',
        marker={"size": 10, "color": 'rgba(0,0,0,0)'},
        hoverinfo='text',
        text=[f"Lat: {lat:.2f}, Lon: {lon:.2f}" for lat, lon in zip(grid_lats.flatten(), grid_lons.flatten())],
        showlegend=False,
        customdata=np.column_stack((grid_lats.flatten(), grid_lons.flatten()))
    ), row=row, col=col)

if compare_mode:
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.05)
    sub1 = get_layer_data(layer1, ds, var, lead)
    sub2 = get_layer_data(layer2, ds, var, lead)
    add_map_traces(fig, sub1, row=1, col=1)
    add_map_traces(fig, sub2, row=1, col=2)
    fig.update_layout(title=f"Compare: {layer1} vs {layer2}")
else:
    fig = make_subplots(rows=1, cols=1)
    if layer == "Individual Models":
        m = st.selectbox("Select Model", ds.model.values)
        sub = get_layer_data(m, ds, var, lead)
    else:
        sub = get_layer_data(layer, ds, var, lead)
    add_map_traces(fig, sub, row=1, col=1)
    fig.update_layout(title=f"{layer} {var.upper()}")

fig.update_layout(
    plot_bgcolor="#0A192F", paper_bgcolor="rgba(0,0,0,0)",
    margin={"l": 0, "r": 0, "t": 30, "b": 0},
    clickmode="event+select"
)
fig.update_xaxes(showgrid=False, visible=False)
fig.update_yaxes(showgrid=False, visible=False, scaleanchor="x", scaleratio=1)
if compare_mode:
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    st.info("💡 Panning is natively synced across both Cartesian maps.")

st.info("💡 Click any point on the map to analyze it in the Point Explorer.")

sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

if sel and len(sel.get("selection", {}).get("points", [])) > 0:
    pt = sel["selection"]["points"][0]
    if "customdata" in pt:
        lat, lon = pt["customdata"][0], pt["customdata"][1]
        st.session_state["selected_latlon"] = (lat, lon)
        st.success(f"Selected Location: {lat:.2f}°N, {lon:.2f}°E")
        st.page_link("pages/05_point_explorer.py", label="Go to Point Explorer for this location", icon="🔍")
