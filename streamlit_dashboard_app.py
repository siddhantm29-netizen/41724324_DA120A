"""
Streamlit dashboard — PJM East Energy Consumption Forecasting
Deliverable for: Energy Consumption Forecasting Using Machine Learning and Deep Learning
Student: Siddhant Santosh Mathapati (41724324)

Run with:
    streamlit run app.py

Expects, in the same directory:
    - pjme_features.csv      (from the preprocessing/EDA notebook)
    - xgb_pjme_model.pkl     (from the modelling notebook)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(page_title="PJME Load Forecast Dashboard", layout="wide")

TARGET = "PJME_MW"
DATA_PATH = "pjme_features.csv"
MODEL_PATH = "xgb_pjme_model.pkl"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    # The index may be saved as a plain integer (YYYYMMDD) rather than a
    # datetime string, so parse_dates=True won't work.  Convert explicitly.
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
    df.index.name = "Datetime"
    df = df.sort_index()          # must be monotonic for label-based slicing
    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("PJM East — Hourly Energy Load Forecast Dashboard")
st.caption("Energy Consumption Forecasting Using Machine Learning and Deep Learning")

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Run the preprocessing notebook first and place the "
             f"output file next to this script.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    st.warning(f"Could not find `{MODEL_PATH}`. Predictions and SHAP explainability will be "
               f"unavailable until you run the modelling notebook and save the model.")
    model_loaded = False

feature_cols = [c for c in df.columns if c not in [TARGET, "rolling_z"]]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

min_date, max_date = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(max_date - pd.Timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

view = df.loc[str(start_date):str(end_date)]

show_predictions = st.sidebar.checkbox("Show model predictions", value=model_loaded, disabled=not model_loaded)
show_anomalies = st.sidebar.checkbox("Highlight potential anomalies (|rolling z| > 4)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** PJM Interconnection LLC hourly load (2002–2018), via "
    "[Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption), CC0."
)

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Load (selected range)", f"{view[TARGET].mean():,.0f} MW")
col2.metric("Peak Load (selected range)", f"{view[TARGET].max():,.0f} MW")
col3.metric("Min Load (selected range)", f"{view[TARGET].min():,.0f} MW")
col4.metric("Records in range", f"{len(view):,}")

# ---------------------------------------------------------------------------
# Main load chart
# ---------------------------------------------------------------------------
st.subheader("Load Over Time")

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=view[TARGET], mode="lines", name="Actual Load (MW)",
                          line=dict(width=1, color="#1f77b4")))

if show_predictions and model_loaded:
    preds = model.predict(view[feature_cols])
    fig.add_trace(go.Scatter(x=view.index, y=preds, mode="lines", name="Model Prediction (MW)",
                              line=dict(width=1, color="#ff7f0e", dash="dot")))

if show_anomalies and "rolling_z" in df.columns:
    anomalies = view[view["rolling_z"].abs() > 4] if "rolling_z" in view.columns else pd.DataFrame()
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[TARGET], mode="markers",
                                  name="Potential anomaly",
                                  marker=dict(color="red", size=6)))

fig.update_layout(height=450, xaxis_title="Datetime", yaxis_title="Load (MW)",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

if show_predictions and model_loaded:
    errors = view[TARGET].values - preds
    st.caption(
        f"Prediction error over selected range — MAE: {np.mean(np.abs(errors)):,.1f} MW, "
        f"RMSE: {np.sqrt(np.mean(errors ** 2)):,.1f} MW"
    )

# ---------------------------------------------------------------------------
# Seasonality tabs
# ---------------------------------------------------------------------------
st.subheader("Seasonality Patterns")
tab1, tab2, tab3 = st.tabs(["By Hour of Day", "By Day of Week", "By Month"])

with tab1:
    hourly_avg = df.groupby(df.index.hour)[TARGET].mean().reset_index()
    hourly_avg.columns = ["Hour", "Average Load (MW)"]
    fig_h = px.line(hourly_avg, x="Hour", y="Average Load (MW)", markers=True)
    st.plotly_chart(fig_h, use_container_width=True)

with tab2:
    dow_avg = df.groupby(df.index.dayofweek)[TARGET].mean().reset_index()
    dow_avg["Day"] = dow_avg["Datetime" if "Datetime" in dow_avg.columns else dow_avg.columns[0]].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    fig_d = px.bar(dow_avg, x="Day", y=TARGET, labels={TARGET: "Average Load (MW)"})
    st.plotly_chart(fig_d, use_container_width=True)

with tab3:
    monthly_avg = df.groupby(df.index.month)[TARGET].mean().reset_index()
    monthly_avg.columns = ["Month", "Average Load (MW)"]
    fig_m = px.line(monthly_avg, x="Month", y="Average Load (MW)", markers=True)
    st.plotly_chart(fig_m, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
st.subheader("Feature Importance")

if model_loaded:
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(importances, x="Importance", y="Feature", orientation="h",
                      title="XGBoost Feature Importance")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("What drives these predictions? (SHAP)"):
        st.markdown(
            "Run the SHAP section of the modelling notebook and export a summary plot image "
            "(e.g. `shap_summary.png`) to embed a richer, per-prediction explanation here. "
            "Feature importance above reflects the model's overall (gain-based) ranking."
        )
else:
    st.info("Load `xgb_pjme_model.pkl` (from the modelling notebook) to see feature importance.")

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
with st.expander("View raw data for selected range"):
    st.dataframe(view[[TARGET] + feature_cols[:8]].tail(500))
