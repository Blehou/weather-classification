import streamlit as st
from model_loader import load_model
from predictor import predict

from preprocessing import (
    load_encoders,
    preprocess_input,
    align_columns
)

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Weather AI",
    page_icon="🌦️",
    layout="wide"
)

# =========================
# LOAD MODEL + ENCODERS
# =========================
model = load_model()
ohe, le = load_encoders()

cat_cols = ["Cloud Cover", "Season", "Location"]
model_features = model.feature_names_in_

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Controls")
    st.write("Weather AI Model")

    st.info("Version 1.0 - ML Weather Classifier")
    st.divider()

# =========================
# HEADER
# =========================
st.title("🌦️ Weather Prediction")
st.caption("Predict weather conditions using Machine Learning")

st.divider()

# =========================
# DASHBOARD LAYOUT
# =========================

col1, col2, col3 = st.columns(3)

# =========================
# COLUMN 1 - WEATHER
# =========================
with col1:
    st.subheader("🌡️ Weather")

    temperature = st.slider("Temperature (°C)", -30, 60, 20)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)

# =========================
# COLUMN 2 - ATMOSPHERE
# =========================
with col2:
    st.subheader("🌫️ Atmosphere")

    precipitation = st.slider("Precipitation (%)", 0, 100, 0)
    pressure = st.number_input("Atmospheric Pressure (hPa)", value=1013)
    uv_index = st.slider("UV Index", 0, 15, 5)
    visibility = st.slider("Visibility (km)", 0, 20, 10)

# =========================
# COLUMN 3 - ENVIRONMENT
# =========================
with col3:
    st.subheader("🌍 Environment")

    cloud_cover = st.selectbox(
        "Cloud Cover",
        ["clear", "partly cloudy", "cloudy", "overcast"]
    )

    season = st.selectbox(
        "Season",
        ["Winter", "Spring", "Summer", "Autumn"]
    )

    location = st.selectbox(
        "Location",
        ["inland", "mountain", "coastal"]
    )

st.divider()

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔮 Predict Weather", use_container_width=True):

    input_data = {
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Precipitation (%)": precipitation,
        "Cloud Cover": cloud_cover,
        "Atmospheric Pressure": pressure,
        "UV Index": uv_index,
        "Season": season,
        "Visibility (km)": visibility,
        "Location": location
    }

    # =========================
    # PREPROCESSING
    # =========================
    df = preprocess_input(
        input_data=input_data,
        ohe=ohe,
        cat_cols=cat_cols
    )

    df = align_columns(df, model_features)

    # =========================
    # PREDICTION
    # =========================
    result = predict(model, df)

    # =========================
    # RESULT CARD
    # =========================
    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background-color:#e0f2fe;
            border:1px solid #262730;
            text-align:center;
            font-size:22px;
            margin-top:20px;">
            🌦️ <b>Prediction:</b> {result}
        </div>
        """,
        unsafe_allow_html=True
    )