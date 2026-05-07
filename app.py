from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Time Series Forecasting API",
    description="Sales Forecasting Backend using XGBoost",
    version="1.0"
)

# Load trained model
model = joblib.load("best_model.pkl")


# Home Route
@app.get("/")
def home():

    return {
        "message": "Forecast API Running Successfully"
    }


# Forecast Route
@app.get("/forecast/{state}")
def forecast(state: str):

    # Sample input features
    sample_data = pd.DataFrame({

        'lag_1': [200],
        'lag_7': [180],
        'lag_30': [150],
        'rolling_mean_7': [190],
        'rolling_std_7': [12],
        'day_of_week': [2],
        'month': [5],
        'holiday_flag': [0]

    })

    # Predict next value
    prediction = model.predict(sample_data)

    # Generate dummy 56-day forecast
    future_forecast = [
        float(prediction[0] + np.random.randint(-10, 10))
        for i in range(56)
    ]

    return {

        "state": state,
        "model": "XGBoost",

        "forecast_days": 56,

        "forecast": future_forecast

    }
