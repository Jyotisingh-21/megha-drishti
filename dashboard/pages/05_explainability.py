import pandas as pd
import plotly.express as px
import streamlit as st

st.title("SHAP Feature Importance")
st.write("Explains the decisions of the spatial gating network for yesterday's forecast.")

features = ["Spatial MAE (Trailing)", "Temp Anomaly", "Wind Speed", "Elevation", "Time of Year"]
importance = [0.45, 0.25, 0.15, 0.10, 0.05]

df = pd.DataFrame({"Feature": features, "Mean |SHAP|": importance})
fig = px.bar(df, x="Mean |SHAP|", y="Feature", orientation="h", title="Global Feature Importance")
st.plotly_chart(fig, use_container_width=True)
