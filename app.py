import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Set layout and styles
st.set_page_config(layout="wide")
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSlider>div>div>div>div {background: #4CAF50;}
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("enhanced_weather.csv")
    model = joblib.load("rainfall_model_v2.pkl")
    return df, model

df, model = load_data()

# Add lat/lon coordinates for cities
city_coords = {
    "Gombe": (10.29, 11.17),
    "Bauchi": (10.31, 9.84),
    "Potiskum": (11.71, 11.08),
    "Yola": (9.21, 12.48)
}
df["lat"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[0])
df["lon"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[1])

# Sidebar controls
st.sidebar.header("⚙️ Prediction Parameters")
max_temp = st.sidebar.slider("Max Temperature (°C)", 20.0, 45.0, 30.0, 0.1)
min_temp = st.sidebar.slider("Min Temperature (°C)", 10.0, 30.0, 20.0, 0.1)
humidity = st.sidebar.slider("Humidity (%)", 30, 100, 60)
cloud_cover_proxy = st.sidebar.slider("Cloud Cover Proxy (%)", 0, 100, 50)
month = st.sidebar.selectbox("Month", range(1,13), 6)

# Title
st.title("🌧️ Advanced Rainfall Prediction System")

col1, col2 = st.columns([2, 1])

# Left panel with charts
with col1:
    st.subheader("🌍 Regional Rainfall Distribution (Avg)")

    # Group average rainfall per city
    map_data = df.groupby('city').agg({
        'precipitation_sum': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()

    # Create scatter geo map
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        hover_name='city',
        size='precipitation_sum',
        color='precipitation_sum',
        color_continuous_scale="Blues",
        size_max=30,
        projection="natural earth",
        title="Average Rainfall per City"
    )
    fig.update_layout(
        geo=dict(scope='africa', showcountries=True, countrycolor="Black")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rainfall trend chart
    st.subheader("📈 Historical Trends")
    trend_data = df.groupby(['year', 'month'])['precipitation_sum'].mean().reset_index()
    trend_fig = px.line(
        trend_data,
        x="month",
        y="precipitation_sum",
        color="year",
        title="Monthly Rainfall Patterns",
        labels={'precipitation_sum': 'Rainfall (mm)', 'month': 'Month'}
    )
    st.plotly_chart(trend_fig, use_container_width=True)

# Right panel with prediction
with col2:
    st.subheader("🔮 Rainfall Forecast")

    # Create input features for prediction
    input_features = pd.DataFrame({
        'temperature_2m_max': [max_temp],
        'temperature_2m_min': [min_temp],
        'relative_humidity_2m': [humidity],
        'cloud_cover_proxy': [cloud_cover_proxy],
        'precipitation_lag1': [0.0],  # default or from previous known data
        'precipitation_lag2': [0.0],  # default or from previous known data
        'is_monsoon': [1 if month in [6,7,8,9] else 0],
        'season': ['Summer' if month in [6,7,8] else 'Spring' if month in [3,4,5] else 'Autumn' if month in [9,10,11] else 'Winter'],
        'temp_rolling_7': [max_temp]  # use same as today's max_temp for approximation
    })
    
    
    if st.button("Predict Rainfall", type="primary"):
        try:
            prediction = model.predict(input_features)[0]

            # Estimate confidence (mocked)
            base_confidence = 0.7
            humidity_factor = humidity / 100 * 0.2
            season_factor = 0.1 if input_features['is_monsoon'][0] else -0.05
            confidence = min(0.95, base_confidence + humidity_factor + season_factor)

            # Display prediction
            st.metric("Predicted Rainfall", f"{prediction:.1f} mm")
            st.progress(int(confidence * 100))
            st.caption(f"Prediction confidence: {confidence * 100:.1f}%")

            # Alert based on rainfall level
            if prediction > 10:
                st.success("🌧️ High rainfall expected – Prepare for potential flooding.")
            elif prediction > 5:
                st.info("🌦️ Moderate rainfall – Good for agriculture.")
            else:
                st.warning("🌤️ Low rainfall – Consider water conservation.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Dataset summary
st.subheader("📊 Dataset Overview")
st.dataframe(df.describe(), use_container_width=True)

# Download button
st.download_button(
    label="Download Processed Data",
    data=df.to_csv(index=False),
    file_name="enhanced_weather_data.csv",
    mime="text/csv"
)
