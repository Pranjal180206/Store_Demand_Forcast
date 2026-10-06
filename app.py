import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
import warnings
import textwrap

# Suppress scikit-learn unpickling version warnings
warnings.filterwarnings("ignore")

# Configure page layout and meta
st.set_page_config(
    page_title="Store Demand Forecaster",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-contrast, clean light frontend styling
custom_css = textwrap.dedent("""
<style>
/* Base typography & color enforcement */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a !important;
}

/* Light Background */
.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Ensure all widget labels, markdown, and captions have dark, readable text */
p, span, label, div {
    color: inherit;
}

.stMarkdown, .stMarkdown p {
    color: #0f172a !important;
}

/* Inputs & Form Controls */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    color: #0f172a !important;
    font-weight: 500 !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Header Banner */
.app-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.app-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    line-height: 1.2;
}

.app-subtitle {
    font-size: 0.95rem;
    color: #475569;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Scenario Buttons: Dark Blue Background (#1e3a8a) with Pure White Text */
div.stButton > button {
    background-color: #1e3a8a !important;
    border: 1px solid #172554 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button * {
    color: #ffffff !important;
    font-weight: 600 !important;
}

div.stButton > button:hover {
    background-color: #172554 !important;
    border-color: #0f172a !important;
}

div.stButton > button:active, div.stButton > button:focus {
    background-color: #0f172a !important;
}

/* Forecast Output Box */
.forecast-container {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 24px;
    margin-top: 10px;
    margin-bottom: 16px;
}

.forecast-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #1d4ed8;
    margin-bottom: 4px;
}

.forecast-value {
    font-size: 3.4rem;
    font-weight: 800;
    color: #1e3a8a;
    line-height: 1.05;
    margin: 6px 0;
}

.forecast-detail {
    font-size: 0.92rem;
    color: #334155;
    font-weight: 500;
}

/* Status Badges */
.status-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-right: 6px;
}

.status-open {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.status-closed {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.status-promo {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

/* Secondary KPI Grid */
.kpi-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
}

.kpi-card-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-card-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 4px;
}

/* Calendar Info Caption */
.calendar-info {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.85rem;
    color: #334155;
    font-weight: 500;
    margin-top: 4px;
    margin-bottom: 12px;
}
</style>
""").strip()

st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_forecast_model():
    """Loads the Random Forest regression model via memory mapping."""
    model_path = os.path.join(os.path.dirname(__file__), "best_sales_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path, mmap_mode="r")


# Header
header_markup = textwrap.dedent("""
<div class="app-header">
    <div class="app-title">Store Demand Forecaster</div>
    <div class="app-subtitle">Forecast the exact number of product units demanded per store using trained regression model.</div>
</div>
""").strip()
st.markdown(header_markup, unsafe_allow_html=True)

# Load Model
try:
    with st.spinner("Loading regression model..."):
        model = load_forecast_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Session State for Quick Scenarios
if "store_id" not in st.session_state:
    st.session_state.store_id = 1
if "customers" not in st.session_state:
    st.session_state.customers = 650
if "promo" not in st.session_state:
    st.session_state.promo = True
if "is_open" not in st.session_state:
    st.session_state.is_open = True
if "school" not in st.session_state:
    st.session_state.school = False
if "date" not in st.session_state:
    st.session_state.date = datetime.date(2015, 7, 31)

# Scenario Shortcuts
st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Quick Scenarios</p>", unsafe_allow_html=True)
sc1, sc2, sc3, _ = st.columns([1, 1, 1, 2])

with sc1:
    if st.button("Peak Promo Day", width="stretch"):
        st.session_state.store_id = 1
        st.session_state.customers = 850
        st.session_state.promo = True
        st.session_state.is_open = True
        st.session_state.school = False
        st.rerun()

with sc2:
    if st.button("Normal Weekday", width="stretch"):
        st.session_state.store_id = 1
        st.session_state.customers = 500
        st.session_state.promo = False
        st.session_state.is_open = True
        st.session_state.school = False
        st.rerun()

with sc3:
    if st.button("School Holiday", width="stretch"):
        st.session_state.store_id = 1
        st.session_state.customers = 950
        st.session_state.promo = True
        st.session_state.is_open = True
        st.session_state.school = True
        st.rerun()

st.write("")

# 2-Column Responsive Layout
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; font-size:1.15rem; color:#0f172a; font-weight:700;'>1. Store and Date Parameters</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            store_val = st.number_input(
                "Store ID",
                min_value=1,
                max_value=2000,
                value=int(st.session_state.store_id),
                step=1,
                help="Store identification number",
            )
        with c2:
            date_val = st.date_input(
                "Forecast Date",
                value=st.session_state.date,
                help="Target forecast date",
            )

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_idx = date_val.weekday()
        day_name = day_names[dow_idx]
        dow_num = dow_idx + 1  # 1=Monday ... 7=Sunday

        st.markdown(
            f"""
            <div class="calendar-info">
                Calendar Analysis: <b>{day_name}</b> (Day {dow_num} of week) &bull; Month: <b>{date_val.strftime('%B')}</b> &bull; Year: <b>{date_val.year}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("<h3 style='font-size:1.15rem; color:#0f172a; font-weight:700;'>2. Operational Conditions</h3>", unsafe_allow_html=True)

        cust_val = st.number_input(
            "Customers (Current Month)",
            min_value=0,
            max_value=25000,
            value=int(st.session_state.customers),
            step=25,
            help="Number of customers for the store in the current month",
        )

        t1, t2, t3 = st.columns(3)
        with t1:
            open_bool = st.toggle("Store Open", value=st.session_state.is_open)
        with t2:
            promo_bool = st.toggle("Promo Active", value=st.session_state.promo)
        with t3:
            school_bool = st.toggle("School Holiday", value=st.session_state.school)

# Feature Construction
open_int = 1 if open_bool else 0
promo_int = 1 if promo_bool else 0
school_int = 1 if school_bool else 0
year_val = date_val.year
month_val = date_val.month
day_val = date_val.day

input_df = pd.DataFrame(
    [
        {
            "Store": int(store_val),
            "DayOfWeek": int(dow_num),
            "Customers": int(cust_val),
            "Open": int(open_int),
            "Promo": int(promo_int),
            "SchoolHoliday": int(school_int),
            "Year": int(year_val),
            "Month": int(month_val),
            "Day": int(day_val),
        }
    ]
)

# Run Inference
if open_int == 0:
    predicted_units = 0.0
else:
    try:
        raw_pred = model.predict(input_df)[0]
        predicted_units = max(0.0, float(raw_pred))
    except Exception as err:
        st.error(f"Inference error: {err}")
        predicted_units = 0.0

# Secondary Metric Calculations
units_per_cust = (predicted_units / cust_val) if cust_val > 0 else 0.0
days_in_month = 31 if month_val in [1, 3, 5, 7, 8, 10, 12] else (28 if month_val == 2 else 30)
monthly_projected = predicted_units * (days_in_month * (0.85 if not promo_int else 1.0))

with col_right:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; font-size:1.15rem; color:#0f172a; font-weight:700;'>Forecast Results</h3>", unsafe_allow_html=True)

        # Status Indicators
        open_badge = '<span class="status-pill status-open">STORE OPEN</span>' if open_int else '<span class="status-pill status-closed">STORE CLOSED</span>'
        promo_badge = '<span class="status-pill status-promo">PROMOTION ACTIVE</span>' if promo_int else ''
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                {open_badge}
                {promo_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Forecast Card
        st.markdown(
            f"""
            <div class="forecast-container">
                <div class="forecast-label">Forecasted Demand Units</div>
                <div class="forecast-value">{int(round(predicted_units)):,}</div>
                <div class="forecast-detail">
                    Exact Forecast: <b>{predicted_units:.2f} units</b> for Store #{store_val} on {date_val.strftime('%d %B %Y')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Secondary KPIs
        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-card-label">Units Per Customer</div>
                    <div class="kpi-card-value">{units_per_cust:.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-card-label">Monthly Run-Rate (~{days_in_month} Days)</div>
                    <div class="kpi-card-value">{int(round(monthly_projected)):,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-card-label">Target Period</div>
                    <div class="kpi-card-value" style="font-size: 1.05rem;">{date_val.strftime('%B %Y')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-card-label">Model Architecture</div>
                    <div class="kpi-card-value" style="font-size: 1.05rem;">Random Forest</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if open_int == 0:
            st.warning("Notice: The store is closed on this date, resulting in 0 product units demanded.")
        elif promo_int:
            st.success("Active promotion increases projected demand and footfall conversion.")
        else:
            st.info("Standard regular pricing active without promotional campaigns.")

# Expandable Raw Data Inspector
with st.expander("View Model Feature Vector"):
    st.dataframe(input_df, width="stretch", hide_index=True)
