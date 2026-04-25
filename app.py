import streamlit as st
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Electricity Predictor",
    page_icon="⚡",
    layout="centered"
)

# -------------------------------
# CUSTOM CSS (BACKGROUND + STYLE)
# -------------------------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: white;
}

/* Title Styling */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Button Styling */
.stButton>button {
    background-color: #00c6ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

/* Success Box */
.success-box {
    background-color: rgba(0, 255, 150, 0.2);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL FILES
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="title">⚡ Electricity Consumption Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict next hour electricity usage using historical patterns</div>', unsafe_allow_html=True)

# -------------------------------
# INPUT SECTION (NO HTML BOX BUG)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("⏰ Hour", 0, 23)
    lag1 = st.number_input("🔁 Previous Hour (lag_1)", min_value=0.0, step=1.0)

with col2:
    day = st.slider("📅 Day", 1, 31)
    lag2 = st.number_input("🔁 2nd Previous Hour (lag_2)", min_value=0.0, step=1.0)

rolling_mean = st.number_input("📊 Rolling Mean (last 3 hours)", min_value=0.0, step=1.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("⚡ Predict Consumption"):

    try:
        input_data = np.array([[hour, day, lag1, lag2, rolling_mean]])

        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)

        prediction = model.predict(input_selected)

        st.markdown(
            f'<div class="success-box">⚡ Predicted Usage: {prediction[0]:.4f} kW</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center;'>Built with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)