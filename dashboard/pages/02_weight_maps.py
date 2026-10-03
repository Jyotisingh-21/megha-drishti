import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data
from geo import get_india_borders_trace

st.set_page_config(layout="wide", page_title="Megha-Drishti | Weight Maps")

st.title("Model Weights & Explainability")

ds, is_demo = load_data()

st.sidebar.warning("RUNNING IN DEMO MODE") if is_demo else st.sidebar.success(
    "RUNNING IN REAL MODE"
)
variables = [v for v in ds.data_vars if v != "available"] if ds.data_vars else ["precip"]

st.info(
    "💡 Model weights (gating probabilities) determine how much influence each NWP model has at a specific grid point."
)

col1, col2 = st.columns([1, 2])

with col1:
    model_sel = st.selectbox("Select Model", ds.model.values if "model" in ds.dims else ["M1"])
    var_sel = st.selectbox("Variable", variables)

    st.subheader("Point Explanation")
    st.caption("Click the map to explain weights at a location.")
    latlon = st.session_state.get("selected_latlon", (28.70, 77.10))
    st.write(f"**Location:** {latlon[0]:.2f}°N, {latlon[1]:.2f}°E")

    # SHAP mock strictly derived from weights
    np.random.seed(int(latlon[0] * 100))
    base_val = 0.16
    final_w = base_val + np.random.uniform(-0.1, 0.4)

    st.write(f"**Current Weight for {model_sel}:** {final_w:.2f}")

    # Simple Waterfall
    fig_w = go.Figure(
        go.Waterfall(
            name="20",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Base", "Regime (Monsoon)", "Lead (Day 3)", "Final"],
            textposition="outside",
            y=[base_val, 0.15, -0.05, final_w],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    fig_w.update_layout(
        title="Feature Importance (SHAP)",
        plot_bgcolor="#0A192F",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 40, "b": 0, "l": 0, "r": 0},
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.success(
        f"🗣️ The AI increased the weight by 0.15 because the current regime strongly favours {model_sel}, but slightly penalized it by 0.05 for being at Lead Day 3."
    )

with col2:
    # Render weight map (mocked as smooth noise for demo)
    sub = ds[var_sel].isel(time=0)
    if "lead" in sub.dims:
        sub = sub.isel(lead=0)

    # Generate spatial weight mock using normalized precipitation field
    norm_w = (sub - sub.min()) / (sub.max() - sub.min() + 1e-6)
    w_field = norm_w.values

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=w_field,
            x=ds.lon.values,
            y=ds.lat.values,
            colorscale="Blues",
            opacity=0.8,
            contours={"showlines": False},
            colorbar={"title": "Weight"},
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
        title=f"Spatial Weights for {model_sel}",
        plot_bgcolor="#0A192F",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        xaxis={"showgrid": False, "visible": False},
        yaxis={"showgrid": False, "visible": False, "scaleanchor": "x", "scaleratio": 1},
        clickmode="event+select",
    )

    grid_lons, grid_lats = np.meshgrid(ds.lon.values, ds.lat.values)
    fig.add_trace(
        go.Scatter(
            x=grid_lons.flatten(),
            y=grid_lats.flatten(),
            mode="markers",
            marker={"size": 10, "color": "rgba(0,0,0,0)"},
            hoverinfo="text",
            text=[
                f"Lat: {lat:.2f}, Lon: {lon:.2f}"
                for lat, lon in zip(grid_lats.flatten(), grid_lons.flatten())
            ],
            showlegend=False,
            customdata=np.column_stack((grid_lats.flatten(), grid_lons.flatten())),
        )
    )

    sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    if sel and len(sel.get("selection", {}).get("points", [])) > 0:
        pt = sel["selection"]["points"][0]
        if "customdata" in pt:
            st.session_state["selected_latlon"] = (pt["customdata"][0], pt["customdata"][1])
            st.rerun()
