import json
import os
from datetime import UTC, datetime, timezone

import streamlit as st


def render_header(is_demo: bool):
    # Read status from latest run report if available
    status = "SUCCESS"
    run_date = None
    report_file = "../data/logs/run_report_latest.json"
    if os.path.exists(report_file):
        try:
            with open(report_file, "r") as f:
                r = json.load(f)
                status = r.get("status", "SUCCESS")
                run_date = r.get("date")
        except Exception:
            pass

    color = "#10B981" if status == "SUCCESS" else "#EF4444"
    mode_badge = (
        "<span style='background: #F59E0B; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;'>DEMO</span>"
        if is_demo
        else "<span style='background: #10B981; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;'>REAL</span>"
    )

    stale_badge = ""
    if run_date:
        try:
            rdt = datetime.strptime(run_date, "%Y-%m-%d").replace(tzinfo=UTC)
            if (datetime.now(UTC) - rdt).days > 2:
                stale_badge = "<span style='background: #EF4444; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;'>STALE</span>"
        except Exception:
            pass

    col_mode, col_lang = st.columns([4, 1])

    with col_lang:
        st.session_state["language"] = st.selectbox(
            "Language / भाषा",
            ["English", "Hindi"],
            index=0 if st.session_state.get("language", "English") == "English" else 1,
            label_visibility="collapsed",
            key="header_lang_toggle",
        )
        if st.session_state["language"] == "Hindi":
            st.warning("Note: Hindi translations need native-speaker review.", icon="⚠️")

    st.markdown(
        f"""
        <div class="header-bar">
            <div class="header-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17.5 19c2.5 0 4.5-2 4.5-4.5a4.5 4.5 0 0 0-4-4.47A7 7 0 0 0 4.3 12.3 4.5 4.5 0 0 0 5 21h12.5"/>
                    <path d="M12 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/>
                </svg>
                Megha-Drishti
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                {stale_badge}
                {mode_badge}
                <div style="display: flex; align-items: center; gap: 6px; font-size: 0.9rem; color: #94A3B8;">
                    Pipeline
                    <div class="status-dot" style="background-color: {color}; box-shadow: 0 0 8px {color};"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; padding: 15px;">
            <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">{title}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #fff; margin: 10px 0;">{value}</div>
            <div style="font-size: 0.8rem; color: #00D2FF;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
