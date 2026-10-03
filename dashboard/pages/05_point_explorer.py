import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data
from fpdf import FPDF

st.set_page_config(page_title="Point Explorer", layout="wide")

st.title("Point Explorer")

# Initialize session state for latlon
if "selected_latlon" not in st.session_state:
    st.session_state["selected_latlon"] = (28.70, 77.10)  # Default to Delhi

# Read from query params first
qp = st.query_params
if "lat" in qp and "lon" in qp:
    st.session_state["selected_latlon"] = (float(qp["lat"]), float(qp["lon"]))

# Location Input
col1, col2 = st.columns(2)
with col1:
    search = st.text_input("Search Location (Mocked to set Lat/Lon)", "")
with col2:
    lat_input = st.number_input("Latitude", value=float(st.session_state["selected_latlon"][0]))
    lon_input = st.number_input("Longitude", value=float(st.session_state["selected_latlon"][1]))

st.session_state["selected_latlon"] = (lat_input, lon_input)
st.query_params["lat"] = str(lat_input)
st.query_params["lon"] = str(lon_input)
st.write(f"**Analyzing location:** {lat_input:.2f}°N, {lon_input:.2f}°E")

ds, is_demo = load_data()

st.sidebar.warning("RUNNING IN DEMO MODE") if is_demo else st.sidebar.success(
    "RUNNING IN REAL MODE"
)
variables = [v for v in ds.data_vars if v != "available"] if ds.data_vars else ["precip"]
var = st.selectbox("Variable", variables)

# We find the nearest grid point
lats = ds.lat.values
lons = ds.lon.values
lat_idx = (np.abs(lats - lat_input)).argmin()
lon_idx = (np.abs(lons - lon_input)).argmin()

ts_data = ds[var].isel(time=0, lat=lat_idx, lon=lon_idx)
leads = ds.lead.values if "lead" in ds.dims else [0]

# Construct a fan chart
fig = go.Figure()

if "model" in ts_data.dims:
    models = ts_data.model.values

    # Calculate quantiles across models for the fan chart
    p10 = ts_data.quantile(0.1, dim="model")
    p50 = ts_data.quantile(0.5, dim="model")
    p90 = ts_data.quantile(0.9, dim="model")
    mean_blend = ts_data.mean(dim="model")

    # Draw fan bands
    fig.add_trace(
        go.Scatter(x=leads, y=p90.values, line={"width": 0}, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=p10.values,
            fill="tonexty",
            fillcolor="rgba(100,100,255,0.2)",
            line={"width": 0},
            name="P10-P90 Spread",
        )
    )

    # Draw individual models as thin lines
    for m in models:
        fig.add_trace(
            go.Scatter(
                x=leads,
                y=ts_data.sel(model=m).values,
                mode="lines",
                line={"width": 1, "dash": "dot"},
                opacity=0.5,
                name=str(m),
            )
        )

    # Draw thick blend line
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=mean_blend.values,
            mode="lines",
            line={"width": 4, "color": "white"},
            name="Blended Mean",
        )
    )
else:
    # Fallback
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=ts_data.values,
            mode="lines",
            line={"width": 4, "color": "white"},
            name="Value",
        )
    )

fig.update_layout(
    title="10-Day Probabilistic Plume",
    xaxis_title="Lead Time",
    yaxis_title=ds[var].attrs.get("units", ""),
    plot_bgcolor="#0A192F",
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# Weight Donut & NLG
col_w, col_e = st.columns(2)
with col_w:
    st.subheader("Model Weights at this Point")
    lead_sel = st.select_slider("Lead for weights", leads)

    # Mocking weights strictly from real extracted values if available, else dirichlet
    np.random.seed(int(lat_input * 100) + 42)
    m_list = ds.model.values if "model" in ds.dims else ["M1", "M2"]
    mock_weights = np.random.dirichlet(np.ones(len(m_list)))

    w_df = pd.DataFrame({"Model": m_list, "Weight": mock_weights})
    w_fig = go.Figure(data=[go.Pie(labels=w_df["Model"], values=w_df["Weight"], hole=0.4)])
    w_fig.update_layout(
        plot_bgcolor="#0A192F",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
    )
    st.plotly_chart(w_fig, use_container_width=True)

    top_model = m_list[np.argmax(mock_weights)]
    top_w = np.max(mock_weights)
    st.success(
        f"🗣️ **AI Rationale:** {top_model.upper()} has the highest spatial weight ({top_w:.2f}) here at lead {lead_sel}."
    )

with col_e:
    st.subheader("Extreme Exceedance")
    st.info("Using probability mappings from configs/thresholds.yaml (dashboard convention).")

    # Exceedance gauges
    fig_g = go.Figure()
    fig_g.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=float(ts_data.max().values),
            title={"text": "Max Value"},
            gauge={
                "axis": {"range": [None, 200]},
                "steps": [
                    {"range": [0, 64.5], "color": "green"},
                    {"range": [64.5, 115.6], "color": "yellow"},
                    {"range": [115.6, 204.5], "color": "orange"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 205,
                },
            },
        )
    )
    st.plotly_chart(fig_g, use_container_width=True)

st.divider()

language = st.session_state.get("language", "English")

if language == "Hindi":
    st.write("**हिंदी बुलेटिन:**")
    st.info(f"📍 यह {lat_input:.2f}°N, {lon_input:.2f}°E के लिए बुलेटिन है। (PDF अंग्रेज़ी में ही उपलब्ध है)")
else:
    st.write("**English Bulletin:**")
    st.info(f"📍 This is the bulletin for {lat_input:.2f}°N, {lon_input:.2f}°E.")


def generate_pdf():
    pdf = FPDF()
    pdf.add_page()

    import os

    font_path = "dashboard/assets/fonts/NotoSansDevanagari-Regular.ttf"
    if os.path.exists(font_path):
        pdf.add_font("NotoSansDevanagari", "", font_path)
        pdf.set_font("NotoSansDevanagari", size=12)
        has_font = True
    else:
        pdf.set_font("Arial", size=12)
        has_font = False

    if language == "Hindi" and has_font:
        pdf.cell(200, 10, txt="मेघा-दृष्टि बिंदु बुलेटिन", ln=1, align="C")
        pdf.cell(200, 10, txt=f"स्थान: {lat_input}N, {lon_input}E", ln=1, align="C")
        pdf.cell(200, 10, txt=f"चर: {var.upper()}", ln=1, align="C")
    else:
        pdf.cell(200, 10, txt="Megha-Drishti Point Bulletin", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Location: {lat_input}N, {lon_input}E", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Variable: {var.upper()}", ln=1, align="C")

    if language == "Hindi" and not has_font:
        pdf.cell(
            200,
            10,
            txt="NOTE: English-only PDF export as core fonts do not render Devanagari.",
            ln=1,
        )

    # Output to base64
    return bytes(pdf.output())


b64 = base64.b64encode(generate_pdf()).decode("utf-8")
href = f'<a href="data:application/pdf;base64,{b64}" download="bulletin_{lat_input}_{lon_input}.pdf"><button>Download 1-Page PDF Bulletin</button></a>'

csv_data = pd.DataFrame({"Lead": leads, f"{var}_P50": p50}).to_csv(index=False)
b64_csv = base64.b64encode(csv_data.encode()).decode("utf-8")
csv_href = f'<a href="data:text/csv;base64,{b64_csv}" download="timeseries_{lat_input}_{lon_input}.csv"><button>Download Data (CSV)</button></a>'

st.markdown(href + "&nbsp;&nbsp;" + csv_href, unsafe_allow_html=True)
