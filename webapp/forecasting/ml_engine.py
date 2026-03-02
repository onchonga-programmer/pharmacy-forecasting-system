"""
ML Engine — Pharmacy Forecasting System
Handles model loading, feature engineering for live forecasts,
loading of pre-computed results for dashboard display, and
inventory optimisation calculations.
"""

import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ── Path configuration ────────────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).resolve().parent.parent   # webapp/
PROJECT_ROOT = WEBAPP_DIR.parent                        # PharmacyForecastingSystem/
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = DATA_DIR / "results"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Drug metadata ─────────────────────────────────────────────────────────────
DRUG_LIST = sorted(["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"])

ATC_MAIN_GROUP_MAP = {"M": 0, "N": 1, "R": 2}

DRUG_DESCRIPTIONS = {
    "M01AB": "Anti-inflammatory – Acetic acid derivatives",
    "M01AE": "Anti-inflammatory – Propionic acid derivatives",
    "N02BA": "Analgesics – Salicylic acid",
    "N02BE": "Analgesics – Paracetamol (High Volume)",
    "N05B":  "Anxiolytics",
    "N05C":  "Hypnotics & Sedatives (Low Volume)",
    "R03":   "Obstructive airway drugs",
    "R06":   "Antihistamines",
}

VOLUME_SEGMENTS = {
    "M01AB": "Medium", "M01AE": "Medium", "N02BA": "Medium",
    "N02BE": "High",   "N05B":  "High",   "N05C":  "Low",
    "R03":   "Medium", "R06":   "Medium",
}

# ── Caches ────────────────────────────────────────────────────────────────────
_model_cache = None
_historical_data_cache = None
_hw_cache: dict = {}          # drug_name → fitted ExponentialSmoothing result

# ── Drugs that default to Holt-Winters (low-volume, stable seasonal demand) ──
HW_PREFERRED_DRUGS = {"N05C"}   # extend if more drugs favour HW in future


# ── Loaders ───────────────────────────────────────────────────────────────────

# ── Holt-Winters helpers ──────────────────────────────────────────────────────

def _fit_holt_winters(drug_name: str):
    """
    Fit an additive Holt-Winters (ETS A,A,A) model on ALL historical data for
    drug_name.  The fitted result is memoised in _hw_cache so subsequent calls
    for the same drug are cheap.

    Returns the fitted HoltWintersResults object, or raises on failure.
    """
    global _hw_cache
    if drug_name in _hw_cache:
        return _hw_cache[drug_name]

    df = load_historical_data()
    drug_df = (
        df[df["drug_name"] == drug_name]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    if len(drug_df) < 104:   # need at least 2 full years
        raise ValueError(
            f"Not enough history for Holt-Winters on {drug_name} "
            f"(need ≥ 104 weeks, found {len(drug_df)})."
        )

    series = drug_df["total_quantity"].astype(float)
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, remove_bias=True)
    _hw_cache[drug_name] = fitted
    return fitted


def generate_hw_forecast(drug_name: str, forecast_date) -> dict:
    """
    Generate a Holt-Winters forecast for drug_name on forecast_date.
    Returns a dict with the same shape as generate_forecast() so callers
    can treat both interchangeably.
    """
    forecast_date = pd.Timestamp(forecast_date)
    df = load_historical_data()
    drug_df = (
        df[df["drug_name"] == drug_name]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )

    # Only use data strictly before the forecast date
    history = drug_df[drug_df["week_start_date"] < forecast_date]
    if len(history) < 104:
        return {"error": (
            f"Not enough history for Holt-Winters on {drug_name} before "
            f"{forecast_date.date()} (need ≥ 104 weeks, found {len(history)})."
        )}

    try:
        series = history["total_quantity"].astype(float)
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, remove_bias=True)
        # Forecast 1 step ahead (the week immediately after history ends)
        fcast = fitted.forecast(1)
        prediction = max(0.0, round(float(fcast.iloc[0]), 2))
    except Exception as exc:
        return {"error": f"Holt-Winters fitting failed for {drug_name}: {exc}"}

    # Historical context (same structure as generate_forecast)
    seasonal_avg = float(
        history[history["week_start_date"].dt.isocalendar().week
                == forecast_date.isocalendar()[1]]["total_quantity"].mean()
    ) if len(history) > 0 else float(history["total_quantity"].mean())
    recent_avg = float(history["total_quantity"].tail(4).mean())

    # Use 2025 per-drug HW MAE as confidence width (fallback to 5.0)
    HW_MAE_BY_DRUG = {
        "M01AB": 7.79, "M01AE": 4.62, "N02BA": 4.84, "N02BE": 45.64,
        "N05B": 13.77, "N05C": 1.95,  "R03": 18.49,  "R06": 7.74,
    }
    hw_mae = HW_MAE_BY_DRUG.get(drug_name, 5.0)

    return {
        "drug":             drug_name,
        "drug_description": DRUG_DESCRIPTIONS.get(drug_name, drug_name),
        "volume_segment":   VOLUME_SEGMENTS.get(drug_name, "Medium"),
        "date":             str(forecast_date.date()),
        "prediction":       prediction,
        "lower_bound":      max(0.0, round(prediction - 2 * hw_mae, 2)),
        "upper_bound":      round(prediction + 2 * hw_mae, 2),
        "seasonal_avg":     round(seasonal_avg, 2),
        "recent_avg":       round(recent_avg, 2),
        "mae_2018":         hw_mae,
        "model_used":       "Holt-Winters",
        "error":            None,
    }


def load_model():
    global _model_cache
    if _model_cache is None:
        path = RESULTS_DIR / "xgboost" / "xgboost_final_model.pkl"
        with open(path, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache


def load_historical_data() -> pd.DataFrame:
    global _historical_data_cache
    if _historical_data_cache is None:
        path = PROCESSED_DIR / "sales_modeling.csv"
        df = pd.read_csv(path, parse_dates=["week_start_date"])
        df = df.sort_values(["drug_name", "week_start_date"]).reset_index(drop=True)
        _historical_data_cache = df
    return _historical_data_cache.copy()


def load_2018_metrics() -> dict:
    path = RESULTS_DIR / "xgboost" / "validation_2018_metrics.json"
    with open(path) as f:
        return json.load(f)


def load_2018_per_drug() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "xgboost" / "xgboost_results.csv")


def load_2019_comparison() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "validation_2019" / "2019_per_drug_comparison.csv")
    # Rename Change_% → Change_Percent: % symbol breaks Django template variable lookup
    df = df.rename(columns={"Change_%": "Change_Percent"})
    return df


def load_2018_predictions() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "xgboost" / "xgboost_predictions.csv",
                     parse_dates=["week_start_date"])
    return df  # columns: drug_name, week_start_date, actual, xgb_pred


def load_2019_predictions() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "validation_2019" / "2019_predictions.csv",
                     parse_dates=["week_start_date"])
    # Normalise column names to match 2018 format
    df = df.rename(columns={"total_quantity": "actual", "predicted": "xgb_pred"})
    return df  # columns: week_start_date, drug_name, actual, xgb_pred, ...


def load_quarterly_analysis() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "validation_2019" / "2019_quarterly_analysis.csv")


# ── 2025 Out-of-Sample Validation Results ─────────────────────────────────────

def load_2025_metrics() -> dict:
    """Return pre-computed overall 2025 test metrics for all three models."""
    return {
        "xgb": {"mae": 14.29, "rmse": 24.85, "mape": 28.54, "wmape": 25.11, "r2": 0.8774},
        "hw":  {"mae": 13.11, "rmse": 23.23, "mape": 29.23, "wmape": 23.02, "r2": 0.8928},
        "sar": {"mae": 68.43, "rmse": 136.38, "mape": 133.43, "wmape": 120.20, "r2": -2.693},
    }


def load_2025_predictions() -> pd.DataFrame:
    """Load 2025 per-week predictions for all three models."""
    df = pd.read_csv(
        RESULTS_DIR / "validation_2025" / "2025_predictions.csv",
        parse_dates=["week_start_date"],
    )
    return df  # columns: week_start_date, drug_name, total_quantity, xgb_pred, hw_pred, sarima_pred, ...


def load_2025_per_drug() -> pd.DataFrame:
    """Load 2025 per-drug metric comparison (XGBoost vs HW vs SARIMA)."""
    return pd.read_csv(RESULTS_DIR / "validation_2025" / "2025_per_drug_comparison.csv")


def load_2025_quarterly() -> pd.DataFrame:
    """Load 2025 quarterly breakdown for all three models."""
    return pd.read_csv(RESULTS_DIR / "validation_2025" / "2025_quarterly_analysis.csv")


def build_2025_comparison() -> pd.DataFrame:
    """
    Build a per-drug comparison table for the 2025 test period using the
    legacy dashboard/drift template column names:

      MAE_2018       → XGBoost 2025 per-drug MAE  (primary model)
      MAE_2019       → Holt-Winters 2025 per-drug MAE  (best competitor)
      MAPE_2019      → XGBoost 2025 WMAPE
      R2_2019        → XGBoost 2025 R²
      Change_Percent → % by which HW MAE differs from XGB MAE
                        (negative = HW better)
      Status         → "Degraded" when XGB MAE > HW MAE by > 20 %
    """
    per_drug = load_2025_per_drug()
    rows = []
    for _, r in per_drug.iterrows():
        xgb_mae = float(r["XGB_MAE"])
        hw_mae  = float(r["HW_MAE"])
        # positive → XGB worse than HW; negative → XGB better
        xgb_vs_hw_pct = (xgb_mae - hw_mae) / hw_mae * 100 if hw_mae > 0 else 0.0
        change        = round((hw_mae - xgb_mae) / xgb_mae * 100, 1) if xgb_mae > 0 else 0.0
        status = "Degraded" if xgb_vs_hw_pct > 20 else "On Track"
        rows.append({
            "Drug":           r["Drug"],
            "Segment":        r.get("Segment", ""),
            "MAE_2018":       round(xgb_mae, 2),   # XGBoost 2025
            "MAE_2019":       round(hw_mae, 2),    # Holt-Winters 2025
            "MAPE_2019":      round(float(r["XGB_WMAPE"]), 2),
            "R2_2019":        round(float(r["XGB_R2"]), 4),
            "Change_Percent": change,
            "Status":         status,
            "Winner":         str(r.get("Winner", "")),
            "HW_R2":          round(float(r.get("HW_R2", 0)), 4),
            "SAR_MAE":        round(float(r.get("SAR_MAE", 0)), 2),
        })
    return pd.DataFrame(rows)


# ── Feature engineering for live forecast ─────────────────────────────────────

def _get_model_feature_names() -> list:
    """Return the ordered list of feature names the model was trained on."""
    model = load_model()
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        return model.get_booster().feature_names


def compute_features_for_date(drug_name: str, forecast_date) -> tuple:
    """
    Build a one-row feature DataFrame for the given drug and forecast date.
    Uses historical sales up to (but not including) forecast_date for lags/rolling.

    Returns: (feature_df, error_message)  — one of these will be None.
    """
    forecast_date = pd.Timestamp(forecast_date)
    df = load_historical_data()

    # Historical rows for this drug, strictly before forecast date
    drug_df = (df[df["drug_name"] == drug_name]
               .sort_values("week_start_date")
               .reset_index(drop=True))
    drug_df = drug_df[drug_df["week_start_date"] < forecast_date]

    if len(drug_df) < 52:
        return None, (
            f"Not enough history for {drug_name} before {forecast_date.date()} "
            f"(need ≥ 52 weeks, found {len(drug_df)})."
        )

    qty = drug_df["total_quantity"].values.astype(float)

    # ── Time ──
    dt      = forecast_date
    month   = int(dt.month)
    quarter = int(dt.quarter)
    iso     = dt.isocalendar()
    week_of_year = int(iso[1])
    day_of_year  = int(dt.timetuple().tm_yday)

    # time_index: cumulative weeks from start of full dataset
    min_date   = df["week_start_date"].min()
    time_index = int((dt - min_date).days / 7)

    # ── Lags ──
    lag_1  = qty[-1]
    lag_2  = qty[-2]
    lag_4  = qty[-4]
    lag_13 = qty[-13]
    lag_52 = qty[-52]

    # ── Rolling ──
    def rstat(arr, n):
        w = arr[-n:]
        return float(np.mean(w)), float(np.std(w, ddof=0)), float(np.min(w)), float(np.max(w))

    rm4,  rs4,  rmin4,  rmax4  = rstat(qty, 4)
    rm13, rs13, rmin13, rmax13 = rstat(qty, 13)
    rm26, rs26, rmin26, rmax26 = rstat(qty, 26)
    rm52, rs52, rmin52, rmax52 = rstat(qty, 52)

    # ── EWM ──
    s = pd.Series(qty)
    ewm4  = float(s.ewm(span=4,  adjust=False).mean().iloc[-1])
    ewm13 = float(s.ewm(span=13, adjust=False).mean().iloc[-1])
    ewm26 = float(s.ewm(span=26, adjust=False).mean().iloc[-1])

    # ── Percent change & difference ──
    pct_change_1w = float((qty[-1] - qty[-2]) / qty[-2] * 100) if qty[-2] != 0 else 0.0
    pct_change_4w = float((qty[-1] - qty[-5]) / qty[-5] * 100) if len(qty) >= 5 and qty[-5] != 0 else 0.0
    diff_1w = float(qty[-1] - qty[-2])
    diff_4w = float(qty[-1] - qty[-5]) if len(qty) >= 5 else 0.0

    # ── Expanding ──
    expanding_mean = float(np.mean(qty))
    expanding_std  = float(np.std(qty, ddof=0))
    ratio_to_expanding_mean = float(qty[-1] / expanding_mean) if expanding_mean != 0 else 1.0

    # ── Drug-level stats (from full history available) ──
    drug_avg_sales = expanding_mean
    drug_std_sales = expanding_std
    drug_min_sales = float(np.min(qty))
    drug_max_sales = float(np.max(qty))
    drug_cv        = drug_std_sales / drug_avg_sales if drug_avg_sales > 0 else 0.0

    # ── Encodings ──
    drug_name_encoded      = DRUG_LIST.index(drug_name) if drug_name in DRUG_LIST else 0
    atc_main_group_encoded = ATC_MAIN_GROUP_MAP.get(drug_name[0].upper(), 0)

    features = {
        "month":          month,
        "quarter":        quarter,
        "day_of_year":    day_of_year,
        "week_of_year":   week_of_year,
        "month_sin":      math.sin(2 * math.pi * month / 12),
        "month_cos":      math.cos(2 * math.pi * month / 12),
        "week_sin":       math.sin(2 * math.pi * week_of_year / 52),
        "week_cos":       math.cos(2 * math.pi * week_of_year / 52),
        "total_quantity_lag_1":  lag_1,
        "total_quantity_lag_2":  lag_2,
        "total_quantity_lag_4":  lag_4,
        "total_quantity_lag_13": lag_13,
        "total_quantity_lag_52": lag_52,
        "total_quantity_rolling_mean_4":  rm4,
        "total_quantity_rolling_std_4":   rs4,
        "total_quantity_rolling_min_4":   rmin4,
        "total_quantity_rolling_max_4":   rmax4,
        "total_quantity_rolling_mean_13": rm13,
        "total_quantity_rolling_std_13":  rs13,
        "total_quantity_rolling_min_13":  rmin13,
        "total_quantity_rolling_max_13":  rmax13,
        "total_quantity_rolling_mean_26": rm26,
        "total_quantity_rolling_std_26":  rs26,
        "total_quantity_rolling_min_26":  rmin26,
        "total_quantity_rolling_max_26":  rmax26,
        "total_quantity_rolling_mean_52": rm52,
        "total_quantity_rolling_std_52":  rs52,
        "total_quantity_rolling_min_52":  rmin52,
        "total_quantity_rolling_max_52":  rmax52,
        "total_quantity_ewm_4":  ewm4,
        "total_quantity_ewm_13": ewm13,
        "total_quantity_ewm_26": ewm26,
        "time_index":               time_index,
        "pct_change_1w":            pct_change_1w,
        "pct_change_4w":            pct_change_4w,
        "diff_1w":                  diff_1w,
        "diff_4w":                  diff_4w,
        "expanding_mean":           expanding_mean,
        "expanding_std":            expanding_std,
        "ratio_to_expanding_mean":  ratio_to_expanding_mean,
        "drug_avg_sales":           drug_avg_sales,
        "drug_std_sales":           drug_std_sales,
        "drug_min_sales":           drug_min_sales,
        "drug_max_sales":           drug_max_sales,
        "drug_cv":                  drug_cv,
        "drug_name_encoded":        drug_name_encoded,
        "atc_main_group_encoded":   atc_main_group_encoded,
    }

    feature_df = pd.DataFrame([features])

    # Reorder to exactly match model's training column order
    expected = _get_model_feature_names()
    if expected:
        try:
            feature_df = feature_df[expected]
        except KeyError as e:
            return None, f"Feature column mismatch: {e}"

    return feature_df, None


def generate_forecast(drug_name: str, forecast_date) -> dict:
    """
    Generate a sales forecast for drug_name on forecast_date.

    Strategy:
      1. If drug_name is in HW_PREFERRED_DRUGS, use Holt-Winters directly.
      2. Otherwise try XGBoost first.
      3. If XGBoost fails for any reason (not enough history, feature
         mismatch, unknown drug …) fall back to Holt-Winters automatically.

    Returns a dict with prediction, bounds, and context info.
    Includes `model_used` key so callers can show which model was used.
    """
    # ── Step 1: route low-volume / HW-preferred drugs directly ───────────────
    if drug_name in HW_PREFERRED_DRUGS:
        result = generate_hw_forecast(drug_name, forecast_date)
        if result.get("error") is None:
            return result
        # HW also failed — fall through to XGBoost attempt below

    # ── Step 2: try XGBoost ───────────────────────────────────────────────────
    try:
        feature_df, err = compute_features_for_date(drug_name, forecast_date)
        if err:
            raise ValueError(err)

        model      = load_model()
        prediction = float(model.predict(feature_df)[0])
        prediction = max(0.0, round(prediction, 2))
        xgb_ok     = True
    except Exception as xgb_exc:
        xgb_ok    = False
        xgb_error = str(xgb_exc)

    # ── Step 3: fall back to Holt-Winters if XGBoost failed ───────────────────
    if not xgb_ok:
        hw_result = generate_hw_forecast(drug_name, forecast_date)
        if hw_result.get("error") is None:
            hw_result["fallback_reason"] = xgb_error
            return hw_result
        # Both models failed
        return {"error": f"XGBoost: {xgb_error} | Holt-Winters: {hw_result['error']}"}

    # Historical context
    df       = load_historical_data()
    drug_df  = (df[df["drug_name"] == drug_name]
                .sort_values("week_start_date"))
    past     = drug_df[drug_df["week_start_date"] < pd.Timestamp(forecast_date)]

    # Seasonal average: same ISO week in previous years
    target_week = pd.Timestamp(forecast_date).isocalendar()[1]
    past["iso_week"] = past["week_start_date"].dt.isocalendar().week
    same_week = past[past["iso_week"] == target_week]["total_quantity"]
    seasonal_avg = float(same_week.mean()) if len(same_week) > 0 else float(past["total_quantity"].mean())
    recent_avg   = float(past["total_quantity"].tail(4).mean())

    # Confidence range: ± 2 × 2018 MAE for this drug
    try:
        per_drug = load_2018_per_drug()
        row      = per_drug[per_drug["Drug"] == drug_name]
        mae_2018 = float(row["MAE"].iloc[0]) if len(row) > 0 else 2.0
    except Exception:
        mae_2018 = 2.0

    return {
        "drug":              drug_name,
        "drug_description":  DRUG_DESCRIPTIONS.get(drug_name, drug_name),
        "volume_segment":    VOLUME_SEGMENTS.get(drug_name, "Medium"),
        "date":              str(pd.Timestamp(forecast_date).date()),
        "prediction":        prediction,
        "lower_bound":       max(0.0, round(prediction - 2 * mae_2018, 2)),
        "upper_bound":       round(prediction + 2 * mae_2018, 2),
        "seasonal_avg":      round(seasonal_avg, 2),
        "recent_avg":        round(recent_avg, 2),
        "mae_2018":          mae_2018,
        "model_used":        "XGBoost",
        "error":             None,
    }


def get_forecast_chart_data(drug_name: str, forecast_date) -> dict:
    """
    Returns 12 weeks of actual past sales and 12 weeks of predicted future
    sales for the sales trend line chart.
    Fits Holt-Winters on all historical data up to forecast_date so the
    predictions are grounded in the correct window.
    """
    try:
        target_ts = pd.Timestamp(forecast_date)
        df        = load_historical_data()
        drug_df   = (
            df[df["drug_name"] == drug_name]
            .sort_values("week_start_date")
            .reset_index(drop=True)
        )

        train_df = drug_df[drug_df["week_start_date"] < target_ts]
        if len(train_df) < 60:
            return {"error": "Not enough historical data to build chart"}

        # ── Past 12 weeks of actual sales ────────────────────────────────────
        past_df    = train_df.tail(12)
        past_dates = [str(d.date()) for d in past_df["week_start_date"]]
        past_sales = [round(float(v), 1) for v in past_df["total_quantity"]]

        # ── Fit HW on all training values, forecast 12 steps ahead ──────────
        series   = train_df["total_quantity"].values.astype(float)
        hw_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
        )
        hw_fit    = hw_model.fit(optimized=True)
        raw_preds = hw_fit.forecast(12)

        future_dates = [
            str((target_ts + pd.Timedelta(weeks=i)).date()) for i in range(12)
        ]
        future_preds = [max(0.0, round(float(v), 1)) for v in raw_preds]
        forecast_4w  = round(sum(future_preds[:4]))

        return {
            "past_dates":   past_dates,
            "past_sales":   past_sales,
            "future_dates": future_dates,
            "future_preds": future_preds,
            "forecast_4w":  forecast_4w,
            "error":        None,
        }
    except Exception as e:
        return {"error": str(e)}


def process_csv_forecast(file_obj) -> list:
    """
    Batch forecast from uploaded CSV (columns: drug_name, datum).
    Returns list of forecast result dicts.
    """
    try:
        df = pd.read_csv(file_obj)
        df.columns = [c.strip().lower() for c in df.columns]
        if "drug_name" not in df.columns or "datum" not in df.columns:
            return [{"error": "CSV must have columns: drug_name, datum"}]
        df["datum"] = pd.to_datetime(df["datum"])
        return [generate_forecast(str(r["drug_name"]).upper(), r["datum"])
                for _, r in df.iterrows()]
    except Exception as e:
        return [{"error": f"Failed to process CSV: {e}"}]


# ── Inventory Optimisation ────────────────────────────────────────────────────

def compute_inventory_recommendation(
    drug_name: str,
    current_stock: float,
    lead_time_weeks: int,
    service_level_pct: float,
    unit_cost: float,
    order_cost: float,
    holding_cost_pct: float,
) -> dict:
    """
    Compute inventory optimisation metrics for a given drug using demand
    statistics derived from historical sales data.

    Parameters
    ----------
    drug_name         : ATC code, e.g. "M01AB"
    current_stock     : units currently on hand
    lead_time_weeks   : weeks from placing order to delivery
    service_level_pct : desired service level (90 / 95 / 98 / 99)
    unit_cost         : purchase cost per unit (£)
    order_cost        : fixed cost per order (£)  — used in EOQ formula
    holding_cost_pct  : annual holding cost as % of unit cost (e.g. 25 = 25 %)

    Returns
    -------
    dict with all computed inventory metrics, or {"error": msg} on failure.
    """
    df = load_historical_data()
    drug_df = (
        df[df["drug_name"] == drug_name]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )

    if len(drug_df) == 0:
        return {"error": f"No historical data found for {drug_name}"}

    qty = drug_df["total_quantity"].values.astype(float)

    # ── Demand statistics ──────────────────────────────────────────────────
    avg_weekly    = float(np.mean(qty))
    std_weekly    = float(np.std(qty, ddof=1))
    annual_demand = avg_weekly * 52.0

    # ── Z-score for desired service level ─────────────────────────────────
    z_map = {90.0: 1.282, 95.0: 1.645, 98.0: 2.054, 99.0: 2.326}
    z = z_map.get(float(service_level_pct), 1.645)

    # ── Safety stock:  SS = Z × σ_demand × √(lead_time) ───────────────────
    safety_stock = max(0.0, round(z * std_weekly * math.sqrt(lead_time_weeks), 1))

    # ── Reorder point: ROP = μ × L + SS ───────────────────────────────────
    demand_during_lt = round(avg_weekly * lead_time_weeks, 1)
    reorder_point    = round(demand_during_lt + safety_stock, 1)

    # ── Economic Order Quantity: EOQ = √(2DS / H) ─────────────────────────
    H = unit_cost * (holding_cost_pct / 100.0)
    if H > 0 and annual_demand > 0 and order_cost > 0:
        eoq = round(math.sqrt((2.0 * annual_demand * order_cost) / H), 0)
    else:
        eoq = None

    # ── Weeks of stock coverage ────────────────────────────────────────────
    weeks_of_coverage = round(current_stock / avg_weekly, 1) if avg_weekly > 0 else 0.0

    # ── Stock status ───────────────────────────────────────────────────────
    if current_stock <= safety_stock:
        status        = "CRITICAL"
        status_colour = "danger"
        status_icon   = "exclamation-octagon-fill"
    elif current_stock <= reorder_point:
        status        = "ORDER NOW"
        status_colour = "warning"
        status_icon   = "cart-fill"
    elif current_stock <= reorder_point * 1.30:
        status        = "APPROACHING"
        status_colour = "info"
        status_icon   = "arrow-down-circle-fill"
    else:
        status        = "ADEQUATE"
        status_colour = "success"
        status_icon   = "check-circle-fill"

    # ── Suggested order quantity ───────────────────────────────────────────
    base_order = eoq if eoq else round(demand_during_lt + safety_stock)
    if status in ("CRITICAL", "ORDER NOW"):
        # Enough to restore stock to ROP + one EOQ cycle above it
        shortfall      = max(0.0, reorder_point - current_stock)
        suggested_order = int(math.ceil(shortfall + base_order))
    else:
        suggested_order = int(base_order)

    # ── Visual gauge markers (as % of display range) ──────────────────────
    max_display = max(reorder_point * 3.0, float(current_stock), 1.0)
    stock_pct   = min(100, round(current_stock    / max_display * 100))
    rop_pct     = min(100, round(reorder_point    / max_display * 100))
    ss_pct      = min(100, round(safety_stock     / max_display * 100))

    return {
        "drug":               drug_name,
        "drug_description":   DRUG_DESCRIPTIONS.get(drug_name, drug_name),
        "avg_weekly_demand":  round(avg_weekly, 1),
        "std_weekly_demand":  round(std_weekly, 1),
        "annual_demand":      round(annual_demand, 1),
        "lead_time_weeks":    lead_time_weeks,
        "service_level_pct":  service_level_pct,
        "z_score":            z,
        "safety_stock":       safety_stock,
        "demand_during_lt":   demand_during_lt,
        "reorder_point":      reorder_point,
        "eoq":                eoq,
        "current_stock":      current_stock,
        "weeks_of_coverage":  weeks_of_coverage,
        "status":             status,
        "status_colour":      status_colour,
        "status_icon":        status_icon,
        "suggested_order":    suggested_order,
        "stock_pct":          stock_pct,
        "rop_pct":            rop_pct,
        "ss_pct":             ss_pct,
        "unit_cost":          unit_cost,
        "order_cost":         order_cost,
        "holding_cost_pct":   holding_cost_pct,
        "order_value":        round(suggested_order * unit_cost, 2),
        "error":              None,
    }
