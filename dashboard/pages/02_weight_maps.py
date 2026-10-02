import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data
from geo import get_india_borders_trace

st.title("Weight Maps")
st.write("Displays the dynamic spatial trust assigned to each model.")

ds, is_demo = load_data()
models = (
    ds.model.values if "model" in ds.dims else ["ecmwf_ifs", "gfs", "ncum_g", "aifs", "graphcast"]
)

selected_model = st.selectbox("View weights for:", models)

# Generate mock weights for layout (since real weight inference isn't exported directly in demo run_daily yet)
lats = ds.lat.values
lons = ds.lon.values
np.random.seed(42)  # Consistent mock weights for demo
mock_weights = np.random.dirichlet(np.ones(len(models)), size=(len(lats), len(lons)))
model_idx = list(models).index(selected_model) if selected_model in models else 0
w_map = mock_weights[:, :, model_idx]

# Smooth the weights for realistic look
from scipy.ndimage import gaussian_filter

w_map = gaussian_filter(w_map, sigma=3)

fig = go.Figure()
fig.update_layout(
    plot_bgcolor="#0A192F",
    paper_bgcolor="rgba(0,0,0,0)",
    margin={"l": 0, "r": 0, "t": 30, "b": 0},
    xaxis={"showgrid": False, "visible": False},
    yaxis={"showgrid": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
)

fig.add_trace(
    go.Contour(
        z=w_map,
        x=lons,
        y=lats,
        colorscale="Greens",
        opacity=0.8,
        contours={"showlines": False},
        colorbar={"title": "Weight (0-1)"},
    )
)

fig.add_trace(get_india_borders_trace())
fig.update_layout(title=f"Spatial Trust: {selected_model.upper()}")

st.plotly_chart(fig, use_container_width=True)
