import streamlit as st

st.title("About Megha-Drishti")

st.markdown("""
**Megha-Drishti** is a Hybrid AI-NWP Forecast Blending System developed for MoES-NCMRWF.

### Geographic Boundary Disclaimer
> **Illustrative boundary for demonstration. For official use, replace with the Survey of India approved boundary.**

### Data Sources
- **Physical NWP**: ECMWF IFS, NOAA GFS, NCUM-G.
- **AI Models**: ECMWF AIFS, GraphCast, Pangu-Weather.

### Pipeline Architecture
1. **Harmonization**: 0.25°/0.5° grid normalisation.
2. **Bias Correction**: Quantile mapping & Lapse rate corrections.
3. **Regimes**: Synoptic clustering (PCA+KMeans).
4. **Blending**: Spatial Mixture-of-Experts neural network.
5. **Calibration**: EMOS scaling for probabilistic bounds.
""")
