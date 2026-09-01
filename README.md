# Energy Consumption Forecasting Using Machine Learning

**Student:** Siddhant Santosh Mathapati
**Student ID:** 41724324
**Programme:** Master of Data Analytics — Data Analytics Project
**Repository:** [github.com/siddhantm29-netizen/41724324_DA120A](https://github.com/siddhantm29-netizen/41724324_DA120A)

---

## Overview

This project forecasts short-to-medium term hourly electricity demand for the PJM East region
using classical statistical baselines, gradient boosting, and a probabilistic model, then
compares them and explains the best-performing model with SHAP.

**Research question:** Can machine learning models significantly outperform classical
statistical baselines in forecasting hourly electricity demand, and which features drive
prediction accuracy?

## Dataset

- **Source:** [Kaggle — robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (CC0 Public Domain)
- **File used:** `PJME_hourly.csv` — PJM East hourly load, 2002–2018
- **Provider:** PJM Interconnection LLC, a regional transmission organization managing the grid
  across 13 US states and Washington, D.C.
- **After cleaning & feature engineering:** 145,224 hourly records

## Repository Structure

```
.
├── README.md                              # This file
├── project_proposal.md / .pdf             # Original project proposal
├── PJME_hourly.csv                        # Raw dataset (from Kaggle)
├── energy_forecasting_eda.ipynb           # Notebook 1: preprocessing, EDA, feature engineering
├── energy_forecasting_modelling.ipynb     # Notebook 2: modelling, evaluation, explainability
├── pjme_features.csv                      # Feature set exported by Notebook 1 (generated)
├── xgb_pjme_model.pkl                     # Trained XGBoost model (generated)
├── streamlit_dashboard_app.py             # Interactive dashboard (deliverable)
├── requirements.txt                       # Dependencies for the Streamlit dashboard
├── energy_forecasting_presentation.pptx   # Project presentation
└── charts/                                # Notebook-generated chart images (used in the presentation)
    ├── train_test_split.png
    ├── sarima_forecast_vs_actual.png
    ├── gradient_boosting_forecast_sample.png
    ├── prophet_components.png
    ├── benchmark_comparison_mae_rmse_mape.png
    ├── model_ranking_by_rmse.png
    ├── improvement_over_naive_baseline.png
    ├── actual_vs_predicted_best_model.png
    ├── residual_diagnostics_best_model.png
    ├── walkforward_cv_rmse_by_fold.png
    ├── shap_summary_plot.png
    ├── shap_feature_importance_bar.png
    └── shap_dependence_top_feature.png
```

## Methodology

1. **Data Preprocessing & EDA** — Parse timestamps, reindex to a complete hourly grid, interpolate
   gaps, flag anomalies via rolling z-score, and explore daily/weekly/monthly/yearly seasonality
   (including STL decomposition).
2. **Feature Engineering** — Calendar features (hour, day-of-week, month, cyclical encodings,
   US holiday flag), lag features (24h, 168h), and 7-day rolling mean/std.
3. **Modelling** — Naive seasonal and SARIMA baselines; XGBoost and LightGBM on engineered
   tabular features; Facebook Prophet with holiday regressors.
4. **Evaluation** — MAE, RMSE, and MAPE on a held-out final-12-months test set, confirmed with
   5-fold walk-forward cross-validation.
5. **Explainability** — SHAP applied to the best-performing model (LightGBM) to identify the
   features driving demand predictions.
6. **Deliverables** — Comparative model analysis, an interactive Streamlit dashboard, a project
   presentation, and this reproducible codebase.

## How to Run

### 1. Environment

For the notebooks:

```bash
pip install pandas numpy matplotlib seaborn statsmodels xgboost lightgbm prophet \
            shap scikit-learn joblib
```

For the Streamlit dashboard only, see `requirements.txt`.

### 2. Run the notebooks in order

1. Open `energy_forecasting_eda.ipynb` in Kaggle (or Jupyter) with `PJME_hourly.csv` available
   at `/kaggle/input/hourly-energy-consumption/PJME_hourly.csv` (or in the working directory).
   Run all cells — this produces `pjme_features.csv`.
2. Open `energy_forecasting_modelling.ipynb` in the same environment, with `pjme_features.csv`
   in the working directory. Run all cells — this produces `xgb_pjme_model.pkl`, the full model
   comparison, and the SHAP analysis.

### 3. Launch the dashboard

```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard_app.py
```

Requires `pjme_features.csv` and `xgb_pjme_model.pkl` from steps above to be in the same
directory as the script.

### 4. Deployment

The dashboard is designed for [Streamlit Community Cloud](https://streamlit.io/cloud): push
this repo (including `pjme_features.csv` and `xgb_pjme_model.pkl`) to GitHub, then deploy
`streamlit_dashboard_app.py` from share.streamlit.io.

## Results Summary

Actual results on the held-out final 12 months (Aug 2017 – Aug 2018), ranked by RMSE:

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| **LightGBM** | 1,907.5 | **2,561.6** | 5.98% |
| XGBoost | 1,913.7 | 2,565.9 | 6.02% |
| Prophet | 3,231.8 | 4,217.7 | 10.20% |
| Naive Seasonal (168h) | 3,581.5 | 4,852.0 | 11.22% |
| SARIMA (daily) | 6,810.0 | 7,619.6 | 23.55% |

**Key findings:**
- LightGBM and XGBoost are effectively tied for best, cutting RMSE by ~47% versus the naive
  seasonal baseline.
- Daily-resampled SARIMA underperforms even the naive hourly baseline on this dataset.
- 5-fold walk-forward cross-validation on XGBoost confirms the result holds across time windows
  (mean RMSE: 2,488.5 MW).
- SHAP identifies `lag_24h` (yesterday's same-hour load) as the dominant driver of predictions,
  followed by day-of-week and hour-of-day.


## License

Dataset: CC0 Public Domain (Kaggle). Code: for academic use as part of the Master of Data
Analytics programme.
