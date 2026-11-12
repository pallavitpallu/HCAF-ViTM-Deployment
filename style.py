
import streamlit as st


def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --bg: #F5F1FA;
        --surface: #FFFFFF;
        --surface-soft: #F0E9FA;
        --purple: #7041B8;
        --purple-dark: #4B287F;
        --lavender: #8B65C5;
        --text: #302342;
        --muted: #756A83;
        --border: #E5D9F2;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text);
    }

    .stApp {
        background: linear-gradient(135deg, #F8F5FC 0%, #F0EAF8 55%, #F7F3FB 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 245, 252, 0.94);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--purple-dark) !important;
        letter-spacing: -0.3px !important;
        font-weight: 700 !important;
    }

    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.12rem !important; }
    h4 { font-size: 1rem !important; }

    p, label, .stCaption, li {
        color: var(--muted);
        font-size: 0.88rem !important;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] [aria-current="page"] {
        background: #EEE5FA;
        border-radius: 9px;
        color: var(--purple-dark) !important;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(74, 40, 127, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--purple-dark) !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }

    .project-hero {
        padding: 30px;
        border: 1px solid #E2D3F4;
        border-radius: 18px;
        background: linear-gradient(120deg, #FFFFFF, #F0E7FB);
        box-shadow: 0 7px 24px rgba(74, 40, 127, 0.07);
        margin-bottom: 22px;
    }

    .project-hero h1,
    .project-hero h2,
    .project-hero h3 {
        color: var(--purple-dark) !important;
    }

    .project-hero p {
        color: #665779 !important;
        font-size: 0.9rem !important;
    }

    .eyebrow, .section-label {
        color: var(--purple) !important;
        font-size: 0.7rem !important;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .project-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 21px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(74, 40, 127, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .project-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(74, 40, 127, 0.10);
    }

    .project-card h3 {
        color: var(--purple-dark) !important;
    }

    .stButton > button {
        background: var(--purple);
        color: #FFFFFF !important;
        border: 1px solid var(--purple);
        border-radius: 9px;
        padding: 0.48rem 1rem;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: var(--purple-dark);
        border-color: var(--purple-dark);
        color: #FFFFFF !important;
    }

    input, textarea {
        background: #FFFFFF !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }

    [data-baseweb="select"] {
        background: #FFFFFF;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--purple-dark) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        background: #FFFFFF;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    a {
        color: var(--purple) !important;
    }

    </style>
    """, unsafe_allow_html=True)
