import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Time Series Forecasting Dashboard")

st.markdown("---")

# Sidebar
st.sidebar.header("Forecast Settings")

# State selection
state = st.sidebar.selectbox(
    "Select State",
    [
        "Karnataka",
        "Tamil Nadu",
        "Kerala",
        "Delhi",
        "Maharashtra"
    ]
)

# Forecast button
generate = st.sidebar.button("Generate Forecast")

# Main content
st.subheader("Sales Forecasting using XGBoost + FastAPI")

st.write(
    """
    This dashboard predicts future sales for different states
    using machine learning forecasting models.
    """
)

# Generate forecast
if generate:

    # API URL
    url = f"http://127.0.0.1:8000/forecast/{state}"

    # Request prediction
    response = requests.get(url)

    # Convert response to JSON
    data = response.json()

    # Extract forecast
    forecast = data["forecast"]

    # Create dataframe
    forecast_df = pd.DataFrame({

        "Day": range(1, len(forecast)+1),

        "Forecast Sales": forecast

    })

    # Display success message
    st.success(f"Forecast generated for {state}")

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Days",
        len(forecast)
    )

    col2.metric(
        "Average Sales",
        round(forecast_df["Forecast Sales"].mean(), 2)
    )

    col3.metric(
        "Maximum Sales",
        round(forecast_df["Forecast Sales"].max(), 2)
    )

    st.markdown("---")

    # Forecast table
    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # Forecast chart
    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        forecast_df["Day"],
        forecast_df["Forecast Sales"],
        marker='o'
    )

    ax.set_xlabel("Days")

    ax.set_ylabel("Sales")

    ax.set_title(f"56-Day Sales Forecast for {state}")

    ax.grid(True)

    st.pyplot(fig)

    # Download CSV
    csv = forecast_df.to_csv(index=False)

    st.download_button(

        label="Download Forecast CSV",

        data=csv,

        file_name=f"{state}_forecast.csv",

        mime="text/csv"

    )

else:

    st.info("Select a state and click Generate Forecast")
