"""
Streamlit dashboard — PJM East Energy Consumption Forecasting
Student: Siddhant Santosh Mathapati (41724324)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import joblib

st.set_page_config(page_title="PJME Load Forecast Dashboard", layout="wide")

TARGET     = "PJME_MW"
DATA_PATH  = "pjme_features.csv"
MODEL_PATH = "xgb_pjme_model.pkl"


# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().sum() > 0 and "hour" in df.columns:
        dates = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
        idx   = dates + pd.to_timedelta(df["hour"], unit="h")
    df.index      = idx
    df.index.name = "Datetime"
    df = df[df.index.notna()].sort_index()
    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def diebold_mariano(actual, pred1, pred2, h=1):
    """Two-sided DM test. Negative stat = pred1 more accurate."""
    actual = np.asarray(actual)
    e1 = actual - np.asarray(pred1)
    e2 = actual - np.asarray(pred2)
    d  = e1**2 - e2**2
    T  = len(d)
    if T < 10:
        return np.nan, np.nan
    d_bar  = d.mean()
    gamma0 = np.var(d, ddof=1)
    nw_sum = sum((1 - k / h) * np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)) if h > 1 else 0
    var_d  = (gamma0 + 2 * nw_sum) / T
    if var_d <= 0:
        return np.nan, np.nan
    stat  = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(abs(stat)))
    return round(float(stat), 4), round(float(p_val), 5)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("PJM East — Hourly Energy Load Forecast Dashboard")
st.caption("Energy Consumption Forecasting Using Machine Learning and Deep Learning")

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Run the preprocessing notebook first.")
    st.stop()

try:
    model        = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    st.warning(f"Could not find `{MODEL_PATH}`. Predictions will be unavailable.")
    model_loaded = False

feature_cols = [c for c in df.columns if c not in [TARGET, "rolling_z"]]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

min_date, max_date = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(max_date - pd.Timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
view = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

show_predictions = st.sidebar.checkbox("Show XGBoost prediction", value=model_loaded, disabled=not model_loaded)
show_naive       = st.sidebar.checkbox("Show Naive Seasonal (168h lag)", value=False,
                                        help="Simplest statistical baseline: same hour last week.")
show_anomalies   = st.sidebar.checkbox("Highlight anomalies (|rolling z| > 4)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** PJM Interconnection LLC hourly load (2002–2018), via "
    "[Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption), CC0."
)

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Load (selected range)",  f"{view[TARGET].mean():,.0f} MW")
c2.metric("Peak Load (selected range)", f"{view[TARGET].max():,.0f} MW")
c3.metric("Min Load (selected range)",  f"{view[TARGET].min():,.0f} MW")
c4.metric("Records in range",           f"{len(view):,}")

# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
st.subheader("Load Over Time")

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=view[TARGET], mode="lines",
                          name="Actual Load (MW)", line=dict(width=1.2, color="#1f77b4")))

preds = None
if show_predictions and model_loaded:
    preds = model.predict(view[feature_cols])
    fig.add_trace(go.Scatter(x=view.index, y=preds, mode="lines",
                              name="XGBoost (ML)", line=dict(width=1.2, color="#22c55e", dash="dot")))

if show_naive and "lag_168h" in view.columns:
    fig.add_trace(go.Scatter(x=view.index, y=view["lag_168h"], mode="lines",
                              name="Naive Seasonal 168h (Statistical)",
                              line=dict(width=1.2, color="#f59e0b", dash="dash")))

if show_anomalies and "rolling_z" in view.columns:
    anoms = view[view["rolling_z"].abs() > 4]
    if not anoms.empty:
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms[TARGET], mode="markers",
                                  name="Anomaly", marker=dict(color="red", size=5)))

fig.update_layout(height=460, xaxis_title="Datetime", yaxis_title="Load (MW)",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

if preds is not None:
    errs  = view[TARGET].values - preds
    nerrs = view[TARGET].values - view["lag_168h"].values if "lag_168h" in view.columns else None
    cap   = (f"**XGBoost** — MAE: {np.mean(np.abs(errs)):,.1f} MW | "
             f"RMSE: {np.sqrt(np.mean(errs**2)):,.1f} MW | "
             f"MAPE: {np.mean(np.abs(errs/view[TARGET].values))*100:.2f}%")
    if nerrs is not None:
        cap += (f"　　**Naive 168h** — MAE: {np.mean(np.abs(nerrs)):,.1f} MW | "
                f"RMSE: {np.sqrt(np.mean(nerrs**2)):,.1f} MW")
    st.caption(cap)

# ---------------------------------------------------------------------------
# ML vs Statistical Comparison
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("ML vs Statistical Model Comparison")
st.markdown("Results on the held-out **final 12 months** (Aug 2017 – Aug 2018). Lower error = better.")

tab_bench, tab_visual, tab_dm = st.tabs([
    "📊 Benchmark Results",
    "📈 Visual Comparison (selected range)",
    "🔬 Statistical Significance (DM Test)",
])

# ── Benchmark table ─────────────────────────────────────────────────────────
with tab_bench:
    bench = pd.DataFrame([
        {"Model": "LightGBM",            "Type": "ML",          "MAE (MW)": 1907.5, "RMSE (MW)": 2561.6, "MAPE (%)": 5.98,  "vs Naive RMSE": "−47.2%"},
        {"Model": "XGBoost",             "Type": "ML",          "MAE (MW)": 1913.7, "RMSE (MW)": 2565.9, "MAPE (%)": 6.02,  "vs Naive RMSE": "−47.1%"},
        {"Model": "Prophet",             "Type": "Probabilistic","MAE (MW)": 3231.8, "RMSE (MW)": 4217.7, "MAPE (%)": 10.20, "vs Naive RMSE": "−13.1%"},
        {"Model": "Naive Seasonal (168h)","Type": "Statistical", "MAE (MW)": 3581.5, "RMSE (MW)": 4852.0, "MAPE (%)": 11.22, "vs Naive RMSE": "— baseline"},
        {"Model": "SARIMA (daily)",      "Type": "Statistical",  "MAE (MW)": 6810.0, "RMSE (MW)": 7619.6, "MAPE (%)": 23.55, "vs Naive RMSE": "+57.1% worse"},
    ])

    def highlight_type(row):
        if row["Type"] == "ML":
            return ["background-color: rgba(34,197,94,0.15)"] * len(row)
        elif row["Type"] == "Statistical" and "SARIMA" in row["Model"]:
            return ["background-color: rgba(239,68,68,0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        bench.style.apply(highlight_type, axis=1)
             .format({"MAE (MW)": "{:,.1f}", "RMSE (MW)": "{:,.1f}", "MAPE (%)": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

    ca, cb = st.columns(2)
    color_map = {"ML": "#22c55e", "Probabilistic": "#f59e0b", "Statistical": "#94a3b3"}
    with ca:
        fig_r = px.bar(bench, x="Model", y="RMSE (MW)", color="Type",
                        color_discrete_map=color_map, title="RMSE by Model", text_auto=".0f")
        fig_r.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)
    with cb:
        fig_m = px.bar(bench, x="Model", y="MAPE (%)", color="Type",
                        color_discrete_map=color_map, title="MAPE by Model", text_auto=".2f")
        fig_m.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

    st.info(
        "🟢 **ML models (LightGBM, XGBoost)** reduce RMSE by ~47% vs the naive seasonal baseline.  \n"
        "🔴 **SARIMA** (fit on daily-resampled data) performs worst — aggregating to daily granularity "
        "destroys the hourly patterns that drive accuracy."
    )

# ── Visual comparison ────────────────────────────────────────────────────────
with tab_visual:
    if "lag_168h" not in view.columns:
        st.warning("lag_168h not found — re-run the EDA notebook to regenerate pjme_features.csv.")
    else:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=view.index, y=view[TARGET], mode="lines",
                                      name="Actual Load", line=dict(width=1.5, color="#1f77b4")))

        _preds = None
        if model_loaded:
            _preds = model.predict(view[feature_cols])
            fig_cmp.add_trace(go.Scatter(x=view.index, y=_preds, mode="lines",
                                          name="XGBoost (ML)", line=dict(width=1.2, color="#22c55e", dash="dot")))

        fig_cmp.add_trace(go.Scatter(x=view.index, y=view["lag_168h"], mode="lines",
                                      name="Naive Seasonal 168h (Statistical)",
                                      line=dict(width=1.2, color="#f59e0b", dash="dash")))

        fig_cmp.update_layout(height=440, title="Actual vs ML vs Statistical — selected range",
                               xaxis_title="Datetime", yaxis_title="Load (MW)",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_cmp, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        naive_rmse = np.sqrt(np.mean((view[TARGET].values - view["lag_168h"].values)**2))
        naive_mae  = np.mean(np.abs(view[TARGET].values - view["lag_168h"].values))
        m3.metric("Naive MAE",  f"{naive_mae:,.1f} MW")
        m4.metric("Naive RMSE", f"{naive_rmse:,.1f} MW")
        if _preds is not None:
            xgb_rmse = np.sqrt(np.mean((view[TARGET].values - _preds)**2))
            xgb_mae  = np.mean(np.abs(view[TARGET].values - _preds))
            m1.metric("XGBoost MAE",  f"{xgb_mae:,.1f} MW")
            m2.metric("XGBoost RMSE", f"{xgb_rmse:,.1f} MW")
            impv = (1 - xgb_rmse / naive_rmse) * 100
            st.success(
                f"On the selected {len(view):,} hours — XGBoost reduces RMSE by **{impv:.1f}%** "
                "over the Naive Seasonal baseline."
            )

# ── DM Test ──────────────────────────────────────────────────────────────────
with tab_dm:
    st.markdown(
        "The **Diebold-Mariano test** checks whether the difference in forecast accuracy "
        "is statistically significant — not just a numerical coincidence.\n\n"
        "- **DM statistic < 0** → challenger is more accurate  \n"
        "- **p-value < 0.05** → difference is statistically significant  \n"
        "- The live test below runs on **your selected date range** (XGBoost vs Naive)"
    )

    if not model_loaded:
        st.warning("Load `xgb_pjme_model.pkl` to run the live DM test.")
    elif "lag_168h" not in view.columns:
        st.warning("lag_168h not found in the feature CSV.")
    elif len(view) < 50:
        st.warning("Select at least 50 hours to run the DM test.")
    else:
        _xgb  = model.predict(view[feature_cols])
        _naive = view["lag_168h"].values
        _act   = view[TARGET].values

        stat, pval = diebold_mariano(_act, _xgb, _naive)

        if np.isnan(stat):
            st.warning("Could not compute DM statistic — try a wider date range.")
        else:
            if   pval < 0.001: sig = "*** p < 0.001"
            elif pval < 0.01:  sig = "**  p < 0.01"
            elif pval < 0.05:  sig = "*   p < 0.05"
            else:              sig = "n.s. p ≥ 0.05"

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("DM Statistic", f"{stat:.4f}",  help="Negative = XGBoost more accurate")
            mc2.metric("p-value",      f"{pval:.5f}")
            mc3.metric("Significance", sig.split()[0])

            if pval < 0.05:
                winner = "XGBoost" if stat < 0 else "Naive Seasonal"
                st.success(
                    f"✅ The DM test **rejects** the null hypothesis (p = {pval:.5f}). "
                    f"**{winner}** is statistically significantly more accurate on the selected range."
                )
            else:
                st.info(
                    f"The DM test **fails to reject** H₀ (p = {pval:.5f}). "
                    "Try expanding the date range for more statistical power."
                )

    st.markdown("#### Full 12-Month Test Set — Hardcoded Results (from modelling notebook)")
    st.caption("Computed on all 8,760 hours of the held-out test set. SARIMA excluded (daily granularity).")
    dm_full = pd.DataFrame([
        {"Challenger": "LightGBM", "Benchmark": "Naive (168h)", "DM Stat": -14.82, "p-value": "<0.001", "Sig.": "***", "Verdict": "LightGBM significantly better"},
        {"Challenger": "XGBoost",  "Benchmark": "Naive (168h)", "DM Stat": -14.79, "p-value": "<0.001", "Sig.": "***", "Verdict": "XGBoost significantly better"},
        {"Challenger": "Prophet",  "Benchmark": "Naive (168h)", "DM Stat": -2.31,  "p-value": "0.021",  "Sig.": "*",   "Verdict": "Prophet significantly better"},
        {"Challenger": "LightGBM", "Benchmark": "XGBoost",      "DM Stat": -0.08,  "p-value": "0.936",  "Sig.": "n.s.","Verdict": "No significant difference"},
    ])
    st.dataframe(dm_full, use_container_width=True, hide_index=True)
    st.info(
        "**Key takeaway for your research question:** LightGBM and XGBoost are statistically "
        "indistinguishable from each other (p = 0.936), but **both are highly significantly "
        "better than the Naive Seasonal baseline** (p < 0.001), directly answering whether ML "
        "outperforms classical statistical methods."
    )

# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Seasonality Patterns")
tab1, tab2, tab3 = st.tabs(["By Hour of Day", "By Day of Week", "By Month"])

with tab1:
    h_avg = df.groupby(df.index.hour)[TARGET].mean().reset_index()
    h_avg.columns = ["Hour", "Average Load (MW)"]
    st.plotly_chart(px.line(h_avg, x="Hour", y="Average Load (MW)", markers=True), use_container_width=True)

with tab2:
    d_avg = df.groupby(df.index.dayofweek)[TARGET].mean().reset_index()
    d_avg.columns = ["DOW", TARGET]
    d_avg["Day"] = d_avg["DOW"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
    st.plotly_chart(px.bar(d_avg, x="Day", y=TARGET, labels={TARGET: "Average Load (MW)"}), use_container_width=True)

with tab3:
    m_avg = df.groupby(df.index.month)[TARGET].mean().reset_index()
    m_avg.columns = ["Month", "Average Load (MW)"]
    st.plotly_chart(px.line(m_avg, x="Month", y="Average Load (MW)", markers=True), use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Feature Importance (XGBoost — Gain-Based)")

if model_loaded:
    imps = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})\
             .sort_values("Importance", ascending=False)
    fig_imp = px.bar(imps, x="Importance", y="Feature", orientation="h",
                      title="XGBoost Feature Importance (gain)")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("Why does lag_24h dominate?"):
        st.markdown(
            "SHAP analysis (modelling notebook) confirms **`lag_24h`** (yesterday's same-hour load) "
            "accounts for ~40% of total SHAP importance. Electricity demand is highly habit-driven — "
            "daily routines repeat, making yesterday's load the strongest single predictor.  \n\n"
            "Next most important: `lag_168h` (same hour last week), `rolling_mean_7d` (local trend), "
            "and `hour` / `hour_sin` (daily cycle)."
        )
else:
    st.info("Load `xgb_pjme_model.pkl` to see feature importance.")

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
with st.expander("View raw data for selected range"):
    st.dataframe(view[[TARGET] + feature_cols[:8]].tail(500))
