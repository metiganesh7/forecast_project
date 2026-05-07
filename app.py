import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Forecast Dashboard",
    layout="wide"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv("Forecasting.csv")

df.columns = df.columns.str.lower()

# Convert date safely
df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
)

df = df.dropna(subset=['date'])

# ----------------------------------
# STATES
# ----------------------------------

states = sorted(df['state'].unique())

# ----------------------------------
# TITLE
# ----------------------------------

st.title("📈 AI Forecast Dashboard")

st.markdown("---")

# ----------------------------------
# SIDEBAR
# ----------------------------------

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

generate = st.sidebar.button(
    "Generate Forecast"
)

# ----------------------------------
# FILTER DATA
# ----------------------------------

state_data = df[
    df['state'] == selected_state
]

# Group data
state_data = state_data.groupby(
    'date'
)['total'].sum().reset_index()

# ----------------------------------
# HISTORICAL GRAPH
# ----------------------------------

st.subheader(
    f"Historical Sales - {selected_state}"
)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    state_data['date'],
    state_data['total']
)

ax.set_xlabel("Date")

ax.set_ylabel("Sales")

ax.grid(True)

st.pyplot(fig)

# ----------------------------------
# FORECAST
# ----------------------------------

if generate:

    url = (
        f"http://127.0.0.1:8000/"
        f"forecast/{selected_state}"
    )

    response = requests.get(url)

    data = response.json()

    # ----------------------------------
    # BEST MODEL
    # ----------------------------------

    st.success(
        f"Best Model: "
        f"{data['best_model']}"
    )

    # ----------------------------------
    # MODEL COMPARISON
    # ----------------------------------

    metrics = data['metrics']

    comparison_df = pd.DataFrame({

        'Model': list(metrics.keys()),

        'RMSE': [

            metrics[m]['RMSE']

            for m in metrics

        ],

        'MAE': [

            metrics[m]['MAE']

            for m in metrics

        ]

    })

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # ----------------------------------
    # RMSE GRAPH
    # ----------------------------------

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.bar(
        comparison_df['Model'],
        comparison_df['RMSE']
    )

    ax2.set_title("RMSE Comparison")

    st.pyplot(fig2)

    # ----------------------------------
    # FORECAST GRAPH
    # ----------------------------------

    forecasts = data['forecasts']

    forecast_df = pd.DataFrame({

        'Day': range(
            1,
            len(
                forecasts['XGBoost']
            ) + 1
        ),

        'SARIMA': forecasts['SARIMA'],

        'Prophet': forecasts['Prophet'],

        'XGBoost': forecasts['XGBoost'],

        'LSTM': forecasts['LSTM']

    })

    st.subheader(
        "Forecast Comparison"
    )

    fig3, ax3 = plt.subplots(
        figsize=(14,5)
    )

    ax3.plot(
        forecast_df['Day'],
        forecast_df['SARIMA'],
        label='SARIMA'
    )

    ax3.plot(
        forecast_df['Day'],
        forecast_df['Prophet'],
        label='Prophet'
    )

    ax3.plot(
        forecast_df['Day'],
        forecast_df['XGBoost'],
        label='XGBoost'
    )

    ax3.plot(
        forecast_df['Day'],
        forecast_df['LSTM'],
        label='LSTM'
    )

    ax3.legend()

    ax3.grid(True)

    st.pyplot(fig3)

    # ----------------------------------
    # FORECAST TABLE
    # ----------------------------------

    st.subheader(
        "Forecast Data"
    )

    st.dataframe(
        forecast_df,
        use_container_width=True
    )
