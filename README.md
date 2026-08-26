# ⚡ Energy Consumption Forecasting

> **Master of Data Analytics — DA120A Project**
> **Student:** Siddhant Santosh Mathapati | **ID:** 41724324

Forecasting hourly electricity consumption using classical statistical models, gradient boosting, and probabilistic models on 10+ years of PJM Interconnection data.

---

## 📌 Project Overview

This project builds and compares multiple forecasting models to predict hourly electricity demand (in Megawatts) for the PJM East region of the United States. The goal is to determine which approach best captures the complex seasonal patterns in energy usage, and to explain which features drive predictions using SHAP.

**Research Question:**
> Can machine learning models significantly outperform classical statistical baselines in forecasting hourly electricity demand, and which features drive prediction accuracy?

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Name** | Hourly Energy Consumption |
| **Source** | PJM Interconnection LLC |
| **Platform** | Kaggle |
| **Link** | [kaggle.com — PJME_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=PJME_hourly.csv) |
| **License** | CC0: Public Domain (free to use) |
| **File Used** | `PJME_hourly.csv` (PJM East region) |
| **Time Range** | 2002-01-01 to 2018-08-03 |
| **Frequency** | Hourly |
| **Total Rows** | ~145,000 |
| **Size** | ~12 MB (zipped) |

### Dataset Columns

| Column | Type | Description |
|---|---|---|
| `Datetime` | datetime | Hourly timestamp (index) |
| `PJME_MW` | float | Electricity consumption in Megawatts |

### Key Patterns in the Data
- 📈 **Daily cycle** — Peak demand at 18:00–20:00, trough at 04:00–05:00
- 📅 **Weekly cycle** — Weekday consumption higher than weekends
- ☀️ **Annual cycle** — Summer peak (cooling load) and winter secondary peak (heating)
- 📉 **Long-term trend** — Gradual decline post-2008 (economic and efficiency effects)

### Download Instructions

```bash
pip install kaggle
kaggle datasets download -d robikscube/hourly-energy-consumption
unzip hourly-energy-consumption.zip -d data/
```

> Requires a free Kaggle account and API token at `~/.kaggle/kaggle.json`

---

## 🗂️ Project Structure

```
41724324_DA120A/
│
├── data/
│   └── PJME_hourly.csv              # Raw dataset (download from Kaggle)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature creation & transformation
│   ├── 03_baseline_models.ipynb     # Naïve & SARIMA models
│   ├── 04_ml_models.ipynb           # XGBoost & LightGBM
│   ├── 05_prophet.ipynb             # Facebook Prophet
│   └── 06_evaluation_shap.ipynb     # Model comparison & SHAP explainability
│
├── src/
│   ├── data_loader.py               # Data loading & cleaning utilities
│   ├── features.py                  # Feature engineering functions
│   ├── models.py                    # Model training wrappers
│   ├── evaluate.py                  # Evaluation metrics (MAE, RMSE, MAPE)
│   └── utils.py                     # Helper functions & plotting
│
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
│
├── reports/
│   └── figures/                     # Saved plots and charts
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🔬 Methodology

### Step 1 — Data Preprocessing & EDA
- Parse datetime index and sort chronologically
- Detect and impute missing timestamps via forward-fill
- Remove duplicate entries
- Plot raw series, STL decomposition (trend + seasonality + residual)
- Visualise hourly and monthly heatmaps

### Step 2 — Feature Engineering

| Feature Type | Features Created |
|---|---|
| **Calendar** | `hour`, `day_of_week`, `month`, `quarter`, `is_weekend` |
| **Holiday** | `is_us_holiday` (via `holidays` library) |
| **Lag** | `lag_24h`, `lag_48h`, `lag_168h` (1 week ago) |
| **Rolling stats** | `rolling_7d_mean`, `rolling_7d_std`, `rolling_30d_mean` |

### Step 3 — Models

| Model | Library | Purpose |
|---|---|---|
| **Naïve Seasonal** | `numpy` | Baseline (last week same hour) |
| **SARIMA** | `statsmodels` | Classical statistical baseline |
| **XGBoost** | `xgboost` | Gradient boosting on tabular features |
| **LightGBM** | `lightgbm` | Fast gradient boosting (primary ML model) |
| **Prophet** | `prophet` | Probabilistic model with holiday support |

### Step 4 — Train / Test Split

```
|--- Training (2002–2016) ---|-- Validation (2017) --|-- Test (2018) --|
```

Walk-forward cross-validation is used to prevent temporal data leakage.

### Step 5 — Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | Mean of absolute errors | Average error in MW |
| **RMSE** | Square root of mean squared errors | Penalises large errors |
| **MAPE** | Mean absolute percentage error | Relative accuracy (%) |

### Step 6 — Explainability (SHAP)
- SHAP TreeExplainer applied to the best-performing model
- Outputs: feature importance bar chart, dependence plots, force plots

### Step 7 — Dashboard (Streamlit)
- Interactive date range selector
- Actual vs. predicted energy consumption plot
- SHAP feature importance visualisation
- Model comparison summary table

---

## 🛠️ Tech Stack

```
Language        Python 3.11
Data            pandas, numpy
Visualisation   matplotlib, seaborn, plotly
Statistics      statsmodels
ML              scikit-learn, xgboost, lightgbm
Forecasting     prophet
Explainability  shap
Dashboard       streamlit
Environment     venv
Version Control Git + GitHub
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/siddhantm29-netizen/41724324_DA120A.git
cd 41724324_DA120A

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (requires Kaggle API token)
kaggle datasets download -d robikscube/hourly-energy-consumption
unzip hourly-energy-consumption.zip -d data/

# 5. Run notebooks in order (01 through 06)

# 6. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📊 Results Summary

> *(To be completed after model training)*

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naïve Seasonal | — | — | — |
| SARIMA | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| Prophet | — | — | — |

---

## 📄 License

This project is for academic purposes. The dataset is licensed under **CC0: Public Domain**.
