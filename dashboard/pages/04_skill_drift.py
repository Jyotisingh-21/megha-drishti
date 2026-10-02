import streamlit as st

st.title("Model Skill and Drift")

try:
    with open("../docs/results_demo.md", "r") as f:
        st.markdown(f.read())
except FileNotFoundError:
    st.warning("docs/results_demo.md not found. Run `scripts/run_benchmark.py` first.")
