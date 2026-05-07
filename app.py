import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load Dataset
df = pd.read_csv("Forecasting.csv")

# Get unique states from dataset
states = sorted(df['State'].unique())

# Dashboard Title
st.title("📈 Sales Forecasting Dashboard")

st.markdown("---")

# Sidebar
st.sidebar.header("Forecast Settings")

# Dynamic state selection
selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

# Forecast button
generate = st.sidebar.button("Generate Forecast")

# Main description
st.subheader("Time Series Forecasting using XGBoost + FastAPI")

st.write(
    """
    This dashboard predicts future sales for Indian states
    using machine learning forecasting models.
    """
)

# Show state-wise historical data
state_data = df[df['State'] == selected_state]

st.subheader(f"Historical Sales Data - {selected_state}")

st.line_chart(state_data['Sales'])

# Generate forecast
if generate:

    # FastAPI URL
    url = f"http://127.0.0.1:8000/forecast/{selected_state}"

    # API request
    response = requests.get(url)

    # JSON response
    data = response.json()

    # Forecast values
    forecast = data['forecast']

    # Forecast dataframe
    forecast_df = pd.DataFrame({

        "Day": range(1, len(forecast)+1),

        "Forecast Sales": forecast

    })

    st.success(f"Forecast generated for {selected_state}")

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Days",
        len(forecast)
    )

    col2.metric(
        "Average Forecast",
        round(forecast_df["Forecast Sales"].mean(), 2)
    )

    col3.metric(
        "Maximum Forecast",
        round(forecast_df["Forecast Sales"].max(), 2)
    )

    st.markdown("---")

    # Forecast table
    st.subheader("Forecast Table")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # Forecast graph
    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        forecast_df["Day"],
        forecast_df["Forecast Sales"],
        marker='o'
    )

    ax.set_xlabel("Days")

    ax.set_ylabel("Sales")

    ax.set_title(
        f"56-Day Forecast for {selected_state}"
    )

    ax.grid(True)

    st.pyplot(fig)

    # Download CSV
    csv = forecast_df.to_csv(index=False)

    st.download_button(

        label="Download Forecast CSV",

        data=csv,

        file_name=f"{selected_state}_forecast.csv",

        mime="text/csv"

    )

else:

    st.info("Select a state and click Generate Forecast")
