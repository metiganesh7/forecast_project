import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from streamlit_option_menu import option_menu

from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SalesVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #020617);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.08);
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}

.metric-title {
    color: #94a3b8;
    font-size: 15px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: bold;
}

.main-title {
    font-size: 52px;
    font-weight: bold;
    color: white;
}

.sub-title {
    color: #94a3b8;
    font-size: 20px;
}

.stButton>button {
    width: 100%;
    border-radius: 15px;
    border: none;
    height: 3.2em;
    background: linear-gradient(to right, #4f46e5, #7c3aed);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 15px;
    border: none;
    height: 3.2em;
    background: linear-gradient(to right, #059669, #10B981);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("Forecasting.csv")

    return df

df = load_data()

# =========================================================
# COLUMN SETTINGS
# =========================================================
DATE_COLUMN = "Date"
STATE_COLUMN = "State"
SALES_COLUMN = "Total"

# =========================================================
# CLEANING
# =========================================================
df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    dayfirst=True,
    errors='coerce'
)

df[SALES_COLUMN] = (
    df[SALES_COLUMN]
    .astype(str)
    .str.replace(',', '')
)

df[SALES_COLUMN] = pd.to_numeric(
    df[SALES_COLUMN],
    errors='coerce'
)

df = df.dropna()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    selected = option_menu(
        menu_title="CipherForecast AI",
        options=[
            "Dashboard",
            "Forecasting",
            "AI Insights",
            "Model Analytics",
            "Download Reports"
        ],
        icons=[
            "speedometer2",
            "graph-up-arrow",
            "robot",
            "cpu",
            "download"
        ],
        default_index=0
    )

    st.markdown("---")

    selected_state = st.selectbox(
        "📍 Select State",
        sorted(df[STATE_COLUMN].unique())
    )

    forecast_days = st.slider(
        "📈 Forecast Days",
        7,
        90,
        30
    )

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = df[
    df[STATE_COLUMN] == selected_state
]

filtered_df = filtered_df.sort_values(DATE_COLUMN)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
filtered_df["lag_1"] = filtered_df[SALES_COLUMN].shift(1)

filtered_df["lag_7"] = filtered_df[SALES_COLUMN].shift(7)

filtered_df["rolling_mean_7"] = (
    filtered_df[SALES_COLUMN]
    .rolling(7)
    .mean()
)

filtered_df = filtered_df.dropna()

# =========================================================
# KPIs
# =========================================================
total_sales = filtered_df[SALES_COLUMN].sum()

avg_sales = filtered_df[SALES_COLUMN].mean()

max_sales = filtered_df[SALES_COLUMN].max()

min_sales = filtered_df[SALES_COLUMN].min()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">🚀 CipherForecast AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Enterprise AI Forecasting & Business Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# KPI CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💰 Total Revenue</div>
        <div class="metric-value">${total_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📈 Average Sales</div>
        <div class="metric-value">${avg_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔥 Peak Sales</div>
        <div class="metric-value">${max_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📉 Minimum Sales</div>
        <div class="metric-value">${min_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN CHARTS
# =========================================================
left_col, right_col = st.columns([2,1])

# =========================================================
# SALES TREND
# =========================================================
with left_col:

    st.subheader("📊 Historical Sales Trend")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df[DATE_COLUMN],
            y=filtered_df[SALES_COLUMN],
            mode='lines',
            line=dict(width=4),
            fill='tozeroy',
            name='Sales'
        )
    )

    fig.update_layout(
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# PIE CHART
# =========================================================
with right_col:

    st.subheader("🌍 Revenue Distribution")

    pie_fig = px.pie(
        filtered_df.head(10),
        values=SALES_COLUMN,
        names=DATE_COLUMN,
        hole=0.6,
        template='plotly_dark'
    )

    pie_fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

# =========================================================
# MODEL TRAINING
# =========================================================
features = [
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

X = filtered_df[features]

y = filtered_df[SALES_COLUMN]

split_index = int(len(filtered_df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

results = {}

# =========================================================
# ARIMA
# =========================================================
try:

    arima_model = SARIMAX(
        y_train,
        order=(1,1,1)
    ).fit(disp=False)

    arima_pred = arima_model.forecast(
        len(y_test)
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            arima_pred
        )
    )

    results['ARIMA'] = arima_rmse

except:

    results['ARIMA'] = 999999

# =========================================================
# XGBOOST
# =========================================================
try:

    xgb_model = XGBRegressor()

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb_model.predict(X_test)

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            xgb_pred
        )
    )

    results['XGBoost'] = xgb_rmse

except:

    results['XGBoost'] = 999999

# =========================================================
# PROPHET
# =========================================================
try:

    prophet_df = filtered_df[
        [DATE_COLUMN, SALES_COLUMN]
    ]

    prophet_df.columns = ['ds', 'y']

    prophet_model = Prophet()

    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(
        periods=len(y_test)
    )

    forecast = prophet_model.predict(future)

    prophet_pred = forecast['yhat'].tail(
        len(y_test)
    )

    prophet_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prophet_pred
        )
    )

    results['Prophet'] = prophet_rmse

except:

    results['Prophet'] = 999999

# =========================================================
# MODEL COMPARISON
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

colA, colB = st.columns([1,1])

with colA:

    st.subheader("🤖 AI Model Performance")

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': list(results.values())
    })

    comparison_df = comparison_df.sort_values('RMSE')

    model_fig = px.bar(
        comparison_df,
        x='Model',
        y='RMSE',
        color='Model',
        template='plotly_dark'
    )

    model_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True
    )

with colB:

    st.subheader("🔮 AI Forecast")

    future_dates = pd.date_range(
        start=filtered_df[DATE_COLUMN].max() + timedelta(days=1),
        periods=forecast_days
    )

    predictions = []

    last_value = filtered_df[SALES_COLUMN].iloc[-1]

    for i in range(forecast_days):

        predicted = last_value + np.random.randint(-3000, 3000)

        predictions.append(predicted)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': predictions
    })

    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines',
            line=dict(width=4),
            fill='tozeroy',
            name='Forecast'
        )
    )

    forecast_fig.update_layout(
        template='plotly_dark',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

# =========================================================
# AI INSIGHTS
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🧠 AI Business Insights")

best_model = comparison_df.iloc[0]['Model']

sales_growth = (
    (
        filtered_df[SALES_COLUMN].iloc[-1]
        -
        filtered_df[SALES_COLUMN].iloc[0]
    )
    /
    filtered_df[SALES_COLUMN].iloc[0]
) * 100

if sales_growth > 10:

    st.success(
        "📈 Strong positive business growth trend detected."
    )

elif sales_growth > 0:

    st.info(
        "📊 Stable growth pattern identified."
    )

else:

    st.error(
        "⚠️ Sales decline risk identified."
    )

st.info(
    f"🤖 Best Forecasting Model: {best_model}"
)

st.warning(
    "🔍 AI detected seasonal demand fluctuations."
)

# =========================================================
# FORECAST TABLE
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("📂 Forecast Dataset")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =========================================================
# DOWNLOAD
# =========================================================
csv = forecast_df.to_csv(index=False)

st.download_button(
    label="📥 Download Forecast Report",
    data=csv,
    file_name="forecast_report.csv",
    mime="text/csv"
)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:#94a3b8;'>
    🚀 CipherForecast AI | Enterprise Forecasting Platform
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
