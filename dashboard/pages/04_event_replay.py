import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Megha-Drishti | Event Replay")

st.title("Event Replay")

st.info(
    "💡 Event replays evaluate how the model performed on historically significant extreme events."
)

# Since we don't have historical data downloaded for real events, we simulate them as required
events = [
    {
        "name": "Synthetic scenario inspired by Cyclone Biparjoy",
        "type": "Cyclone",
        "date": "Mock Date 1",
        "real": False,
    },
    {
        "name": "Synthetic scenario inspired by Wayanad Landslides",
        "type": "Heavy Rain",
        "date": "Mock Date 2",
        "real": False,
    },
    {
        "name": "Synthetic scenario inspired by Delhi Heatwave",
        "type": "Heatwave",
        "date": "Mock Date 3",
        "real": False,
    },
]

for event in events:
    with st.expander(event["name"]):
        st.error("DEMO: Scenario evaluation (synthetic)")
        st.write(f"**Type:** {event['type']}")

        # Display mock performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw Best Model (RMSE)", "15.2", "Baseline", delta_color="off")
        with col2:
            st.metric("Blended Model (RMSE)", "9.1", "-40%", delta_color="inverse")

        st.caption(
            "Note: This is generated from synthetic fields and does NOT reflect actual predictions for this historical event."
        )
