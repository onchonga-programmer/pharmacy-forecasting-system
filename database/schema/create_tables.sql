-- Create schemas (folders) for organization
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS cleaned_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ============================================
-- RAW DATA SCHEMA (Original, untouched data)
-- ============================================

-- Drug Categories (reference data - in public schema is fine)
CREATE TABLE IF NOT EXISTS public.atc_categories (
    atc_code VARCHAR(10) PRIMARY KEY,
    atc_name VARCHAR(255) NOT NULL,
    category_level VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw Drugs Table
CREATE TABLE IF NOT EXISTS raw_data.drugs_raw (
    drug_id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255) NOT NULL,
    atc_code VARCHAR(10) REFERENCES public.atc_categories(atc_code),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_drug_raw UNIQUE (drug_name, atc_code)
);

-- Raw Weekly Sales Table (exactly as loaded from CSV)
CREATE TABLE IF NOT EXISTS raw_data.sales_weekly_raw (
    week_id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    drug_id INTEGER NOT NULL REFERENCES raw_data.drugs_raw(drug_id),
    total_quantity INTEGER NOT NULL,
    avg_daily_quantity NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_week_drug_raw UNIQUE (year, week_number, drug_id)
);

CREATE INDEX idx_sales_raw_date ON raw_data.sales_weekly_raw(week_start_date);
CREATE INDEX idx_sales_raw_drug ON raw_data.sales_weekly_raw(drug_id);

-- ============================================
-- CLEANED DATA SCHEMA (After data cleaning)
-- ============================================

-- Cleaned Drugs Table
CREATE TABLE IF NOT EXISTS cleaned_data.drugs_clean (
    drug_id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255) NOT NULL,
    atc_code VARCHAR(10) REFERENCES public.atc_categories(atc_code),
    original_drug_id INTEGER REFERENCES raw_data.drugs_raw(drug_id),
    is_active BOOLEAN DEFAULT TRUE,
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_drug_clean UNIQUE (drug_name, atc_code)
);

-- Cleaned Weekly Sales Table (cleaned, no outliers, filled missing values)
CREATE TABLE IF NOT EXISTS cleaned_data.sales_weekly_clean (
    week_id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    drug_id INTEGER NOT NULL REFERENCES cleaned_data.drugs_clean(drug_id),
    total_quantity INTEGER NOT NULL,
    avg_daily_quantity NUMERIC(10,2),
    is_outlier BOOLEAN DEFAULT FALSE,
    was_imputed BOOLEAN DEFAULT FALSE,  -- TRUE if value was filled
    cleaning_notes TEXT,
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_week_drug_clean UNIQUE (year, week_number, drug_id)
);

CREATE INDEX idx_sales_clean_date ON cleaned_data.sales_weekly_clean(week_start_date);
CREATE INDEX idx_sales_clean_drug ON cleaned_data.sales_weekly_clean(drug_id);
CREATE INDEX idx_sales_clean_outlier ON cleaned_data.sales_weekly_clean(is_outlier);

-- ============================================
-- ANALYTICS SCHEMA (Results, forecasts, metrics)
-- ============================================

-- Forecasts Table
CREATE TABLE IF NOT EXISTS analytics.forecasts (
    forecast_id SERIAL PRIMARY KEY,
    drug_id INTEGER NOT NULL REFERENCES cleaned_data.drugs_clean(drug_id),
    forecast_week_start DATE NOT NULL,
    year INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    predicted_quantity NUMERIC(10,2) NOT NULL,
    confidence_lower NUMERIC(10,2),
    confidence_upper NUMERIC(10,2),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_forecast UNIQUE (drug_id, year, week_number, model_name)
);

CREATE INDEX idx_forecasts_date ON analytics.forecasts(forecast_week_start);
CREATE INDEX idx_forecasts_drug ON analytics.forecasts(drug_id);

-- Model Performance Metrics
CREATE TABLE IF NOT EXISTS analytics.model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    drug_id INTEGER REFERENCES cleaned_data.drugs_clean(drug_id),
    metric_type VARCHAR(50),  -- 'RMSE', 'MAE', 'MAPE'
    metric_value NUMERIC(12,4),
    evaluation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);