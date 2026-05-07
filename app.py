import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="Time Series Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------
# LOAD DATASET
# ----------------------------------------

df = pd.read_csv("Forecasting.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# ----------------------------------------
# CHECK REQUIRED COLUMNS
# ----------------------------------------

required_columns = ['state', 'date', 'total']

for col in required_columns:

    if col not in df.columns:

        st.error(f"Missing column in dataset: {col}")

        st.stop()

# ----------------------------------------
# DATE CONVERSION
# ----------------------------------------

df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
)

# Remove invalid dates
df = df.dropna(subset=['date'])

# ----------------------------------------
# SORT DATA
# ----------------------------------------

df = df.sort_values('date')

# ----------------------------------------
# GET STATES
# ----------------------------------------

states = sorted(df['state'].unique())

# ----------------------------------------
# TITLE
# ----------------------------------------

st.title("📈 Time Series Forecasting Dashboard")

st.markdown("---")

# ----------------------------------------
# SIDEBAR
# ----------------------------------------

st.sidebar.header("Forecast Settings")

selected_state = st.sidebar.selectbox(
    "Select Indian State",
    states
)

generate_forecast = st.sidebar.button(
    "Generate Forecast"
)

# ----------------------------------------
# PROJECT DESCRIPTION
# ----------------------------------------

st.subheader(
    "Sales Forecasting using XGBoost + FastAPI"
)

st.write(
    """
    This dashboard predicts future sales for Indian states
    using machine learning forecasting models.
    """
)

# ----------------------------------------
# FILTER STATE DATA
# ----------------------------------------

state_data = df[
    df['state'] == selected_state
]

# ----------------------------------------
# HISTORICAL DATA
# ----------------------------------------

st.subheader(
    f"Historical Sales Data - {selected_state}"
)

historical_df = state_data[
    ['date', 'total']
].copy()

historical_df = historical_df.set_index('date')

st.line_chart(
    historical_df['total']
)

# ----------------------------------------
# SHOW HISTORICAL TABLE
# ----------------------------------------

with st.expander(
    "View Historical Data"
):

    st.dataframe(
        historical_df,
        use_container_width=True
    )

# ----------------------------------------
# FORECAST SECTION
# ----------------------------------------

if generate_forecast:

    try:

        # ----------------------------------------
        # API URL
        # ----------------------------------------

        url = (
            f"http://127.0.0.1:8000/"
            f"forecast/{selected_state}"
        )

        # ----------------------------------------
        # API REQUEST
        # ----------------------------------------

        response = requests.get(url)

        # ----------------------------------------
        # CHECK STATUS
        # ----------------------------------------

        if response.status_code != 200:

            st.error(
                "Failed to get forecast from API"
            )

        else:

            # ----------------------------------------
            # JSON RESPONSE
            # ----------------------------------------

            data = response.json()

            forecast = data['forecast']

            # ----------------------------------------
            # FORECAST DATAFRAME
            # ----------------------------------------

            forecast_df = pd.DataFrame({

                "Day": range(
                    1,
                    len(forecast)+1
                ),

                "Forecast Sales": forecast

            })

            # ----------------------------------------
            # SUCCESS MESSAGE
            # ----------------------------------------

            st.success(
                f"Forecast generated for "
                f"{selected_state}"
            )

            st.markdown("---")

            # ----------------------------------------
            # KPI METRICS
            # ----------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Forecast Days",
                len(forecast)
            )

            col2.metric(
                "Average Forecast",
                round(
                    forecast_df[
                        "Forecast Sales"
                    ].mean(),
                    2
                )
            )

            col3.metric(
                "Maximum Forecast",
                round(
                    forecast_df[
                        "Forecast Sales"
                    ].max(),
                    2
                )
            )

            st.markdown("---")

            # ----------------------------------------
            # FORECAST TABLE
            # ----------------------------------------

            st.subheader(
                "Forecast Table"
            )

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # ----------------------------------------
            # FORECAST CHART
            # ----------------------------------------

            st.subheader(
                "Forecast Visualization"
            )

            fig, ax = plt.subplots(
                figsize=(12,5)
            )

            ax.plot(
                forecast_df["Day"],
                forecast_df[
                    "Forecast Sales"
                ],
                marker='o'
            )

            ax.set_xlabel(
                "Forecast Days"
            )

            ax.set_ylabel(
                "Sales Forecast"
            )

            ax.set_title(
                f"56-Day Forecast "
                f"for {selected_state}"
            )

            ax.grid(True)

            st.pyplot(fig)

            # ----------------------------------------
            # DOWNLOAD CSV
            # ----------------------------------------

            csv = forecast_df.to_csv(
                index=False
            )

            st.download_button(

                label="Download Forecast CSV",

                data=csv,

                file_name=(
                    f"{selected_state}"
                    f"_forecast.csv"
                ),

                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Forecast Error: {e}"
        )

# ----------------------------------------
# DEFAULT MESSAGE
# ----------------------------------------

else:

    st.info(
        "Select a state and click "
        "'Generate Forecast'"
    )
