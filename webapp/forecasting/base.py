

from datetime import date, timedelta
import io
import csv

# ── Page metadata (used in templates & views) ─────────────────────────────────

PAGES = {
    "dashboard": {
        "title":       "Dashboard",
        "icon":        "bi-speedometer2",
        "description": "Overview of model performance and key metrics",
    },
    "results": {
        "title":       "Results & Charts",
        "icon":        "bi-graph-up-arrow",
        "description": "Actual vs Predicted charts for 2018 and 2019",
    },
    "forecast": {
        "title":       "Live Forecast",
        "icon":        "bi-magic",
        "description": "Generate demand forecasts for any drug and date",
    },
    "drift": {
        "title":       "Drift Monitor",
        "icon":        "bi-exclamation-triangle",
        "description": "Model drift analysis — 2018 validation vs 2019 test",
    },
}

# ── Drug metadata ─────────────────────────────────────────────────────────────

DRUG_LABELS = {
    "M01AB": "M01AB — Anti-inflammatory (Acetic acid)",
    "M01AE": "M01AE — Anti-inflammatory (Propionic acid)",
    "N02BA": "N02BA — Analgesics (Salicylic acid)",
    "N02BE": "N02BE — Analgesics (Paracetamol) ⚠ High Drift",
    "N05B":  "N05B  — Anxiolytics",
    "N05C":  "N05C  — Hypnotics & Sedatives",
    "R03":   "R03   — Obstructive airway drugs",
    "R06":   "R06   — Antihistamines",
}

ATC_MAIN_GROUPS = {
    "M": "Musculo-skeletal",
    "N": "Nervous system",
    "R": "Respiratory",
}

# Drift risk levels per drug based on 2018→2019 analysis
DRUG_DRIFT_RISK = {
    "M01AB": "medium",   # +587%
    "M01AE": "medium",   # +289%
    "N02BA": "medium",   # +204%
    "N02BE": "critical",  # +174929%
    "N05B":  "medium",   # +235%
    "N05C":  "low",      # improved in 2019
    "R03":   "low",      # +17%
    "R06":   "medium",   # +394%
}

DRIFT_RISK_BADGES = {
    "low":      ("bg-success",   "Low Risk"),
    "medium":   ("bg-warning text-dark", "Medium Risk"),
    "critical": ("bg-danger",    "Critical"),
}

# ── Date helpers ──────────────────────────────────────────────────────────────

def get_next_monday(from_date: date = None) -> date:
    """Return the next Monday from a given date (defaults to today)."""
    d = from_date or date.today()
    days_ahead = 0 - d.weekday()   # Monday = 0
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def date_to_week_label(d) -> str:
    """Format a date as 'Week 12, 2019' style label."""
    import pandas as pd
    ts = pd.Timestamp(d)
    week = ts.isocalendar()[1]
    return f"Week {week}, {ts.year}"


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string, e.g. 5.6 → '5.6%'"""
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(value: float, decimals: int = 2) -> str:
    """Format a float with fixed decimal places."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_change(value: float) -> dict:
    """
    Return a dict with formatted change string and CSS class.
    e.g. +982.2% → {"text": "+982.2%", "css": "text-danger"}
    """
    try:
        v = float(value)
        if v > 0:
            return {"text": f"+{v:.1f}%", "css": "text-danger"}
        else:
            return {"text": f"{v:.1f}%", "css": "text-success"}
    except (TypeError, ValueError):
        return {"text": "—", "css": "text-muted"}


# ── Drift evaluation ──────────────────────────────────────────────────────────

def evaluate_drift(mae_current: float, mae_baseline: float,
                   threshold_multiplier: float = 1.20) -> dict:
    """
    Evaluate whether a model's current MAE indicates drift.

    Args:
        mae_current:          Current period MAE
        mae_baseline:         Baseline (training/validation) MAE
        threshold_multiplier: Alert threshold multiplier (default 1.20 = 20% above baseline)

    Returns:
        dict with keys: drifted (bool), threshold, pct_change, severity
    """
    threshold  = mae_baseline * threshold_multiplier
    pct_change = ((mae_current - mae_baseline) / mae_baseline * 100) if mae_baseline > 0 else 0.0
    drifted    = mae_current > threshold

    if not drifted:
        severity = "stable"
    elif pct_change < 100:
        severity = "low"
    elif pct_change < 500:
        severity = "medium"
    else:
        severity = "critical"

    return {
        "drifted":    drifted,
        "threshold":  round(threshold, 3),
        "pct_change": round(pct_change, 1),
        "severity":   severity,
    }


# ── CSV template generator ────────────────────────────────────────────────────

SAMPLE_CSV_ROWS = [
    {"drug_name": "M01AB", "datum": "2020-03-02"},
    {"drug_name": "N02BE", "datum": "2020-03-02"},
    {"drug_name": "R03",   "datum": "2020-03-02"},
]


def generate_sample_csv() -> str:
    """
    Return a sample CSV string that users can download as a template
    for batch forecast uploads.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["drug_name", "datum"])
    writer.writeheader()
    for row in SAMPLE_CSV_ROWS:
        writer.writerow(row)
    return output.getvalue()


def generate_sample_csv_bytes() -> bytes:
    """Return the sample CSV as bytes (for Django FileResponse)."""
    return generate_sample_csv().encode("utf-8")


# ── Model performance thresholds ──────────────────────────────────────────────

# Based on 2018 validation results (mae=1.74, mape=5.6, r2=0.9922)
PERFORMANCE_TIERS = {
    "excellent": {"max_mape": 5.0,  "min_r2": 0.95, "label": "Excellent",  "css": "text-success"},
    "good":      {"max_mape": 10.0, "min_r2": 0.85, "label": "Good",       "css": "text-primary"},
    "fair":      {"max_mape": 20.0, "min_r2": 0.70, "label": "Fair",       "css": "text-warning"},
    "poor":      {"max_mape": 999,  "min_r2": -999,  "label": "Poor",      "css": "text-danger"},
}


def get_performance_tier(mape: float, r2: float) -> dict:
    """Classify model performance into a tier based on MAPE and R²."""
    for tier, bounds in PERFORMANCE_TIERS.items():
        if mape <= bounds["max_mape"] and r2 >= bounds["min_r2"]:
            return {"tier": tier, **bounds}
    return {"tier": "poor", **PERFORMANCE_TIERS["poor"]}
