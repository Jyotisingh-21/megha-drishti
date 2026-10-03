import streamlit as st

st.title("About Megha-Drishti")

st.markdown("""
**Megha-Drishti** is a Hybrid AI-NWP Forecast Blending System developed for MoES-NCMRWF.

### Geographic Boundary Disclaimer
> **Illustrative boundary for demonstration. For official use, replace with the Survey of India approved boundary.**

### Data Sources and Attribution
- **Physical NWP**: ECMWF IFS (Contains modified ECMWF information published under a CC-BY-4.0 licence), NOAA GFS (via NOAA Big Data Program), NCUM-G.
- **AI Models**: ECMWF AIFS.
- **Truth Data**: Copernicus ARCO-ERA5 (Contains modified Copernicus Climate Change Service information, CC-BY-4.0), IMD (Data provided by the India Meteorological Department).
- **Boundaries**: DataMeet India State Boundaries (CC-BY-2.5-IN).
See `docs/DATA_LICENCES.md` for full licensing details.

### Pipeline Architecture
1. **Harmonization**: 0.25°/0.5° grid normalisation.
2. **Bias Correction**: Quantile mapping & Lapse rate corrections.
3. **Regimes**: Synoptic clustering (PCA+KMeans).
4. **Blending**: Spatial Mixture-of-Experts neural network.
5. **Calibration**: EMOS scaling for probabilistic bounds.
""")
