import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Page Config ---
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f7f4ef;
        color: #1a1a1a;
    }
    .main { background-color: #f7f4ef; }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 20px;
        padding: 40px 36px 32px 36px;
        margin-bottom: 32px;
        color: white;
    }
    .hero h1 { font-size: 2.2rem; margin-bottom: 6px; color: white; }
    .hero p  { color: #aab4c8; font-size: 1rem; margin: 0; }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #0f3460;
        border-left: 4px solid #e94560;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    .result-box {
        background: linear-gradient(135deg, #0f3460, #e94560);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        color: white;
        margin-top: 24px;
    }
    .result-box .label { font-size: 0.95rem; opacity: 0.85; margin-bottom: 8px; }
    .result-box .price { font-family: 'DM Serif Display', serif; font-size: 3rem; }

    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1.05rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 500;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
<div class="hero">
    <h1>🏠 California House Price Predictor</h1>
    <p>Enter the details below to get an instant price estimate using a Random Forest model trained on real California housing data.</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline

model, pipeline = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `main.py` first to train and save the model.")
    st.stop()

# --- Input Form ---
st.markdown('<div class="section-title">📍 Location</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=-119.0, min_value=-124.0, max_value=-114.0, step=0.01)
with col2:
    latitude = st.number_input("Latitude", value=36.0, min_value=32.0, max_value=42.0, step=0.01)

ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

st.markdown('<div class="section-title">🏘️ Housing Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    housing_median_age = st.slider("Housing Median Age (years)", 1, 52, 20)
    total_rooms = st.number_input("Total Rooms", value=2000, min_value=1, max_value=40000)
    total_bedrooms = st.number_input("Total Bedrooms", value=400, min_value=1, max_value=7000)
with col4:
    population = st.number_input("Population", value=1000, min_value=1, max_value=40000)
    households = st.number_input("Households", value=350, min_value=1, max_value=7000)
    median_income = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5, step=0.1)

# --- Predict ---
st.markdown("###")
if st.button("🔮 Predict House Price"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed = pipeline.transform(input_df)
    prediction = model.predict(transformed)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Median House Value</div>
        <div class="price">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br><center style='color:#aaa;font-size:0.8rem'>Built with Streamlit · Random Forest · California Housing Dataset</center>", unsafe_allow_html=True)
