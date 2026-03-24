# Pharmacy Forecasting System

A  project for forecasting weekly pharmacy demand across 8 ATC drug categories using time-series and machine learning models.

The project includes:
- A full data science workflow (cleaning, EDA, feature engineering, model training, validation)
- Model comparison across XGBoost, Holt-Winters, and SARIMA
- Out-of-sample validation on 2025 (fully unseen test period)
- A Django web application for forecast visualization and decision support

## Project Goals

- Predict weekly drug quantities for inventory planning
- Compare classical and machine learning forecasting approaches
- Evaluate model generalization on unseen time periods
- Provide a usable interface for pharmacy staff

## Models Used

- Baselines: Naive, Seasonal Naive, SES, moving-average variants, Holt-Winters
- SARIMA: per-drug seasonal ARIMA modeling
- XGBoost: supervised learning with engineered lag, rolling, and calendar features

## Dataset

- Source: `data/raw/salesweekly.csv`
- Granularity: weekly sales
- Drug categories: 8 ATC codes (`M01AB`, `M01AE`, `N02BA`, `N02BE`, `N05B`, `N05C`, `R03`, `R06`)
- Extended history and validation artifacts are stored in `data/processed/` and `data/results/`

## Repository Structure

```text
PharmacyForecastingSystem/
|-- config/                  # Config modules (database and environment)
|-- data/
|   |-- raw/                 # Original source data
|   |-- processed/           # Cleaned/feature datasets and train-test splits
|   `-- results/             # Model outputs and validation reports
|-- database/
|   |-- schema/              # SQL schema files
|   `-- seed/                # Seed scripts
|-- documentation/           # Full technical write-up
|-- notebooks/               # End-to-end experiment notebooks
|-- src/data_processing/     # Data loading and cleaning scripts
`-- webapp/                  # Django app for dashboard and forecasts
```

## Key Outputs

- Processed datasets in `data/processed/`
- Model artifacts and predictions in:
	- `data/results/xgboost/`
	- `data/results/baseline_models/`
	- `data/results/sarima/`
- Final 2025 validation summaries in:
	- `data/results/validation_2025/`

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Note: if Django is not available in your environment after installing requirements, install it manually:

```bash
pip install django
```

### 3. Run the Django web application

```bash
cd webapp
python manage.py migrate
python manage.py runserver
```

Open in browser:
- http://127.0.0.1:8000/

Main pages:
- Dashboard
- Results
- Forecast
- Drift
- Inventory

## Notebook Workflow (Recommended Order)

Run notebooks in this order for reproducibility:

1. `notebooks/data_cleaning.ipynb`
2. `notebooks/data_validation.ipynb`
3. `notebooks/pharmacy_sales_eda.ipynb`
4. `notebooks/pharmacy_sales_feature_engineering.ipynb`
5. `notebooks/pharmacy_sales_baseline_models.ipynb`
6. `notebooks/pharmacy_sarima_model.ipynb`
7. `notebooks/pharmacy_xgboost_model.ipynb`
8. `notebooks/model_validation.ipynb`

## Validation Scope

The project uses strict temporal validation to reduce leakage:
- **Training period**: 2014–2024 (10 full years of historical data)
- **Test period**: 2025 (fully out-of-sample, unseen during training)
- Detailed per-drug and quarterly performance comparisons in `data/results/validation_2025/`

## Production-Oriented Notes

- Forecast logic for the web app lives in `webapp/forecasting/ml_engine.py`
- Web settings are in `webapp/pharmacy_system/settings.py`
- Current web app database is SQLite (`webapp/db.sqlite3`)

## Documentation

For the full technical report and development history, see:
- `documentation/project_documentation.txt`


