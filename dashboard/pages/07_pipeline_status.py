import glob
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Megha-Drishti | Pipeline Status")

st.title("Pipeline Status")

reports = sorted(glob.glob("data/logs/run_report_*.json"))
if not reports:
    st.warning("No pipeline runs found.")
else:
    data = []
    for r in reports[-10:]:
        with open(r, "r") as f:
            js = json.load(f)
            data.append(
                {
                    "Date": js.get("date"),
                    "Mode": js.get("mode"),
                    "Status": js.get("status"),
                    "Duration (s)": js.get("total_time_sec"),
                    "Errors": len(js.get("errors", [])),
                    "Warnings": len(js.get("warnings", [])),
                }
            )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.subheader("Latest Run Details")
    with open(reports[-1], "r") as f:
        st.json(json.load(f))
