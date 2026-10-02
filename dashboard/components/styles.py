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
        
        /* Fade in animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .block-container {
            padding-top: 2rem !important;
            animation: fadeIn 0.4s ease-out forwards;
        }

        /* Metric Cards Hover */
        [data-testid="stMetric"] {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            padding: 10px;
            border-radius: 8px;
            background: rgba(21, 31, 50, 0.4);
            border: 1px solid rgba(255,255,255,0.05);
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 210, 255, 0.1);
            background: rgba(21, 31, 50, 0.7);
        }

        /* Number count-up fallback (visual only since we can't inject JS easily) */
        [data-testid="stMetricValue"] {
            animation: fadeIn 0.8s ease-out;
            color: #E2E8F0;
        }

        /* Accessibility: High contrast colors for text */
        p, h1, h2, h3, h4, h5, h6, span {
            color: #F8FAFC;
        }

        /* Prefers Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
