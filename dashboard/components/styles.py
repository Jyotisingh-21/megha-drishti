import streamlit as st


def apply_global_styles():
    st.markdown(
        """
        <style>
        /* Glassmorphism cards */
        .glass-card {
            background: rgba(21, 31, 50, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Model colors */
        :root {
            --color-ecmwf: #3B82F6;
            --color-gfs: #EF4444;
            --color-ncum: #10B981;
            --color-aifs: #8B5CF6;
            --color-graphcast: #F59E0B;
            --color-pangu: #EC4899;
        }
        
        /* Remove default main padding */
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* Modern font stack */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Header bar */
        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: #151F32;
            border-bottom: 1px solid #1E293B;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.2rem;
            font-weight: 600;
            color: #00D2FF;
        }
        
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #10B981;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
