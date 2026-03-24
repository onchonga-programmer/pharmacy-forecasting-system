

import json
import traceback

import pandas as pd
import plotly.graph_objects as go
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from . import ml_engine


# ── Helper ────────────────────────────────────────────────────────────────────

def _fig_json(fig) -> str:
    """Serialise a Plotly figure to a JSON string safe for template embedding."""
    return fig.to_json()


# ── Dashboard ─────────────────────────────────────────────────────────────────

def dashboard(request):
    try:
        m25    = ml_engine.load_2025_metrics()
        cmp_df = ml_engine.build_2025_comparison()

        xgb = m25["xgb"]
        hw  = m25["hw"]

        # Overall 2025 metrics (XGBoost = primary model, HW = best competitor)
        mae_xgb   = xgb["mae"]
        mape_xgb  = xgb["mape"]
        r2_xgb    = xgb["r2"]
        mae_hw    = hw["mae"]
        mape_hw   = hw["mape"]

        # % difference: HW vs XGBoost (negative = HW is more accurate on raw MAE)
        pct_change  = round(((mae_hw - mae_xgb) / mae_xgb) * 100, 1)

        drugs_drifted = int(cmp_df["Status"].str.contains("Degraded", na=False).sum())
        drugs_stable  = len(cmp_df) - drugs_drifted

        # ── MAE comparison bar chart — XGBoost vs HW ──
        fig = go.Figure(data=[
            go.Bar(
                name="Primary Method (2025)",
                x=cmp_df["Drug"].tolist(),
                y=cmp_df["MAE_2018"].tolist(),      # MAE_2018 = XGB_MAE
                marker_color="#0d6efd",
                text=[f"{v:.2f}" for v in cmp_df["MAE_2018"]],
                textposition="outside",
            ),
            go.Bar(
                name="Backup Method (2025)",
                x=cmp_df["Drug"].tolist(),
                y=cmp_df["MAE_2019"].tolist(),      # MAE_2019 = HW_MAE
                marker_color="#198754",
                text=[f"{v:.2f}" for v in cmp_df["MAE_2019"]],
                textposition="outside",
            ),
        ])
        fig.update_layout(
            barmode="group",
            title=dict(text="Forecast Error by Drug — 2025 Test", font=dict(size=14)),
            xaxis_title="Drug (ATC Code)",
            yaxis_title="Average Weekly Error (units)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        mae_threshold = round(mae_xgb * 1.20, 2)

        context = {
            "mae_2018":      mae_xgb,          # Primary model (XGBoost) MAE on 2025 test
            "mape_2018":     mape_xgb,         # Primary model MAPE on 2025 test
            "r2_2018":       r2_xgb,           # Primary model R² on 2025 test
            "mae_2019":      mae_hw,           # Backup model (Holt-Winters) MAE on 2025 test
            "mape_2019":     mape_hw,          # Backup model MAPE on 2025 test
            "r2_2019":       hw["r2"],         # Backup model R² on 2025 test
            "pct_change":    pct_change,       # % difference HW vs XGB on 2025 test
            "mae_threshold": mae_threshold,
            "drugs_drifted": drugs_drifted,
            "drugs_stable":  drugs_stable,
            "total_drugs":   len(cmp_df),
            "mae_chart":     _fig_json(fig),
            "comparison":    cmp_df.to_dict("records"),
            "page":          "dashboard",
        }
    except Exception as e:
        context = {"error": str(e) + "\n\n" + traceback.format_exc(), "page": "dashboard"}

    return render(request, "forecasting/dashboard.html", context)


# ── Results ───────────────────────────────────────────────────────────────────

def results(request):
    selected_drug = request.GET.get("drug", "M01AB")
    selected_year = request.GET.get("year", "2025")

    try:
        if selected_year == "2025":
            preds      = ml_engine.load_2025_predictions()
            drug_preds = preds[preds["drug_name"] == selected_drug].sort_values("week_start_date")

            dates    = drug_preds["week_start_date"].dt.strftime("%Y-%m-%d").tolist()
            actual   = drug_preds["total_quantity"].tolist()
            xgb_pred = drug_preds["xgb_pred"].tolist()
            hw_pred  = drug_preds["hw_pred"].tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=actual,
                mode="lines+markers", name="Actual Sales",
                line=dict(color="#212529", width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=xgb_pred,
                mode="lines+markers", name="Forecast",
                line=dict(color="#0d6efd", width=2, dash="dash"), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=hw_pred,
                mode="lines+markers", name="Backup Method",
                line=dict(color="#198754", width=2, dash="dot"), marker=dict(size=4),
            ))
            fig.update_layout(
                title=dict(
                    text=f"{selected_drug} — Actual Sales vs Forecast (2025)",
                    font=dict(size=15),
                ),
                xaxis_title="Week",
                yaxis_title="Sales Quantity (units)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode="x unified",
            )

            per_drug = ml_engine.load_2025_per_drug()
            row = per_drug[per_drug["Drug"] == selected_drug]
            if len(row) > 0:
                r = row.iloc[0]
                metrics = {
                    "MAE":    round(float(r["XGB_MAE"]), 2),
                    "MAPE":   round(float(r["XGB_WMAPE"]), 2),
                    "R2":     round(float(r["XGB_R2"]), 4),
                    "HW_MAE": round(float(r["HW_MAE"]), 2),
                    "HW_R2":  round(float(r["HW_R2"]), 4),
                    "Winner": str(r.get("Winner", "")),
                }
            else:
                metrics = {}

        elif selected_year == "2018":
            preds      = ml_engine.load_2018_predictions()
            drug_preds = preds[preds["drug_name"] == selected_drug].sort_values("week_start_date")
            dates     = drug_preds["week_start_date"].dt.strftime("%Y-%m-%d").tolist()
            actual    = drug_preds["actual"].tolist()
            predicted = drug_preds["xgb_pred"].tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers",
                name="Actual Sales", line=dict(color="#0d6efd", width=2), marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines+markers",
                name="Forecast", line=dict(color="#fd7e14", width=2, dash="dash"),
                marker=dict(size=5)))
            fig.update_layout(
                title=dict(text=f"{selected_drug} — Actual vs Forecast (2018)", font=dict(size=15)),
                xaxis_title="Week", yaxis_title="Sales Quantity (units)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420, margin=dict(l=40, r=40, t=60, b=40), hovermode="x unified",
            )
            metrics_df = ml_engine.load_2018_per_drug()
            row = metrics_df[metrics_df["Drug"] == selected_drug]
            metrics = row.iloc[0].to_dict() if len(row) > 0 else {}

        else:  # default to 2025 if invalid year selected
            selected_year = "2025"
            preds      = ml_engine.load_2025_predictions()
            drug_preds = preds[preds["drug_name"] == selected_drug].sort_values("week_start_date")
            dates    = drug_preds["week_start_date"].dt.strftime("%Y-%m-%d").tolist()
            actual   = drug_preds["total_quantity"].tolist()
            xgb_pred = drug_preds["xgb_pred"].tolist()
            hw_pred  = drug_preds["hw_pred"].tolist()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=actual,
                mode="lines+markers", name="Actual Sales",
                line=dict(color="#212529", width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=xgb_pred,
                mode="lines+markers", name="Forecast",
                line=dict(color="#0d6efd", width=2, dash="dash"), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=hw_pred,
                mode="lines+markers", name="Backup Method",
                line=dict(color="#198754", width=2, dash="dot"), marker=dict(size=4),
            ))
            fig.update_layout(
                title=dict(text=f"{selected_drug} — Actual Sales vs Forecast (2025)", font=dict(size=15)),
                xaxis_title="Week", yaxis_title="Sales Quantity (units)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420, margin=dict(l=40, r=40, t=60, b=40), hovermode="x unified",
            )
            per_drug = ml_engine.load_2025_per_drug()
            row = per_drug[per_drug["Drug"] == selected_drug]
            if len(row) > 0:
                r = row.iloc[0]
                metrics = {
                    "MAE":    round(float(r["XGB_MAE"]), 2),
                    "MAPE":   round(float(r["XGB_WMAPE"]), 2),
                    "R2":     round(float(r["XGB_R2"]), 4),
                    "HW_MAE": round(float(r["HW_MAE"]), 2),
                    "HW_R2":  round(float(r["HW_R2"]), 4),
                    "Winner": str(r.get("Winner", "")),
                }
            else:
                metrics = {}

        context = {
            "selected_drug": selected_drug,
            "selected_year": selected_year,
            "drug_list":     ml_engine.DRUG_LIST,
            "year_options":  [("2025", "2025 Out-of-Sample Test"), ("2018", "2018 Validation")],
            "chart_json":    _fig_json(fig),
            "metrics":       metrics,
            "page":          "results",
        }
    except Exception as e:
        context = {
            "error":         str(e) + "\n\n" + traceback.format_exc(),
            "selected_drug": selected_drug,
            "selected_year": selected_year,
            "drug_list":     ml_engine.DRUG_LIST,
            "year_options":  [("2025", "2025 Out-of-Sample Test"), ("2018", "2018 Validation")],
            "page":          "results",
        }

    return render(request, "forecasting/results.html", context)


# ── Live Forecast ─────────────────────────────────────────────────────────────

def forecast(request):
    result             = None
    batch_results      = None
    form_data          = {}
    forecast_chart_json = None
    forecast_4w_total   = None

    if request.method == "POST":
        action = request.POST.get("action", "single")

        if action == "single":
            drug_name     = request.POST.get("drug_name", "").strip().upper()
            forecast_date = request.POST.get("forecast_date", "").strip()
            form_data     = {"drug_name": drug_name, "forecast_date": forecast_date}

            if drug_name and forecast_date:
                result = ml_engine.generate_forecast(drug_name, forecast_date)

                # Build 12-week trend chart when single forecast succeeds
                if result and not result.get("error"):
                    _cd = ml_engine.get_forecast_chart_data(drug_name, forecast_date)
                    if not _cd.get("error"):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=_cd["past_dates"],
                            y=_cd["past_sales"],
                            name="Past Sales",
                            mode="lines+markers",
                            line=dict(color="#0284c7", width=2),
                            marker=dict(size=5, color="#0284c7"),
                        ))
                        fig.add_trace(go.Scatter(
                            x=_cd["future_dates"],
                            y=_cd["future_preds"],
                            name="Expected Sales",
                            mode="lines+markers",
                            line=dict(color="#0284c7", width=2, dash="dot"),
                            marker=dict(size=5, color="#0284c7", symbol="circle-open"),
                        ))
                        fig.update_layout(
                            xaxis_title="Week",
                            yaxis_title="Units Sold",
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1),
                            height=280,
                            margin=dict(l=40, r=30, t=30, b=40),
                            hovermode="x unified",
                            shapes=[dict(
                                type="line", xref="x", yref="paper",
                                x0=_cd["future_dates"][0],
                                x1=_cd["future_dates"][0],
                                y0=0, y1=1,
                                line=dict(color="#94a3b8", width=1, dash="dash"),
                            )],
                        )
                        forecast_chart_json = _fig_json(fig)
                        forecast_4w_total   = _cd["forecast_4w"]
            else:
                result = {"error": "Please select a drug and a forecast date."}

        elif action == "csv":
            csv_file = request.FILES.get("csv_file")
            if csv_file:
                batch_results = ml_engine.process_csv_forecast(csv_file)
            else:
                result = {"error": "Please upload a CSV file."}

    context = {
        "result":             result,
        "batch_results":      batch_results,
        "drug_list":          ml_engine.DRUG_LIST,
        "drug_descriptions":  ml_engine.DRUG_DESCRIPTIONS,
        "drug_options":       [(code, ml_engine.DRUG_DESCRIPTIONS.get(code, code)) for code in ml_engine.DRUG_LIST],
        "form_data":          form_data,
        "forecast_chart_json": forecast_chart_json,
        "forecast_4w_total":   forecast_4w_total,
        "page":               "forecast",
    }
    return render(request, "forecasting/forecast.html", context)


# ── Drift Monitor ─────────────────────────────────────────────────────────────

def drift(request):
    try:
        cmp_df  = ml_engine.build_2025_comparison()
        m25     = ml_engine.load_2025_metrics()
        xgb     = m25["xgb"]
        hw      = m25["hw"]

        # Threshold: XGBoost MAE + 20 % buffer
        mae_threshold  = round(xgb["mae"] * 1.20, 2)
        mape_threshold = round(xgb["wmape"] * 1.20, 2)

        drugs_drifted = int(cmp_df["Status"].str.contains("Degraded", na=False).sum())
        drugs_stable  = len(cmp_df) - drugs_drifted

        cmp_sorted = cmp_df.sort_values("MAE_2019", ascending=True)  # sort by HW MAE
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Main Forecast Method",
            y=cmp_sorted["Drug"].tolist(),
            x=cmp_sorted["MAE_2018"].tolist(),      # MAE_2018 = XGB_MAE
            orientation="h",
            marker_color="#0284c7",
            text=[f"{v:.2f}" for v in cmp_sorted["MAE_2018"]],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Backup Forecast Method",
            y=cmp_sorted["Drug"].tolist(),
            x=cmp_sorted["MAE_2019"].tolist(),      # MAE_2019 = HW_MAE
            orientation="h",
            marker_color="#38bdf8",
            text=[f"{v:.2f}" for v in cmp_sorted["MAE_2019"]],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            title=dict(text="Prediction Accuracy by Drug — 2025", font=dict(size=14)),
            xaxis_title="Average Weekly Prediction Error (units) — lower is better",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=460,
            margin=dict(l=70, r=80, t=60, b=40),
        )

        context = {
            "comparison":     cmp_df.to_dict("records"),
            "mae_2018":       xgb["mae"],       # Primary model (XGBoost) MAE on 2025 test
            "mape_2018":      xgb["wmape"],     # Primary model WMAPE on 2025 test
            "r2_2018":        xgb["r2"],        # Primary model R² on 2025 test
            "mae_2019":       hw["mae"],        # Backup model (HW) MAE on 2025 test
            "r2_2019":        hw["r2"],         # Backup model R² on 2025 test
            "mae_threshold":  mae_threshold,
            "mape_threshold": mape_threshold,
            "drugs_drifted":  drugs_drifted,
            "drugs_stable":   drugs_stable,
            "total_drugs":    len(cmp_df),
            "drift_chart":    _fig_json(fig),
            "r2_display_pct": round(xgb["r2"] * 100, 1),
            "page":           "drift",
        }
    except Exception as e:
        context = {"error": str(e) + "\n\n" + traceback.format_exc(), "page": "drift"}

    return render(request, "forecasting/drift.html", context)


# ── Sample CSV download ───────────────────────────────────────────────────────

def sample_csv(request):
    from django.http import HttpResponse
    from .base import generate_sample_csv_bytes
    content = generate_sample_csv_bytes()
    response = HttpResponse(content, content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="forecast_template.csv"'
    return response


# ── Inventory Optimisation ────────────────────────────────────────────────────

def inventory(request):
    result    = None
    form_data = {
        "drug_name":        "M01AB",
        "current_stock":    "500",
        "lead_time_weeks":  "2",
        "service_level":    "95",
        "unit_cost":        "10.00",
        "order_cost":       "50.00",
        "holding_cost_pct": "25",
    }

    if request.method == "POST":
        drug_name        = request.POST.get("drug_name", "").strip().upper()
        current_stock    = request.POST.get("current_stock", "0")
        lead_time_weeks  = request.POST.get("lead_time_weeks", "2")
        service_level    = request.POST.get("service_level", "95")
        unit_cost        = request.POST.get("unit_cost", "10")
        order_cost       = request.POST.get("order_cost", "50")
        holding_cost_pct = request.POST.get("holding_cost_pct", "25")

        form_data = {
            "drug_name":        drug_name,
            "current_stock":    current_stock,
            "lead_time_weeks":  lead_time_weeks,
            "service_level":    service_level,
            "unit_cost":        unit_cost,
            "order_cost":       order_cost,
            "holding_cost_pct": holding_cost_pct,
        }

        try:
            result = ml_engine.compute_inventory_recommendation(
                drug_name         = drug_name,
                current_stock     = float(current_stock),
                lead_time_weeks   = int(lead_time_weeks),
                service_level_pct = float(service_level),
                unit_cost         = float(unit_cost),
                order_cost        = float(order_cost),
                holding_cost_pct  = float(holding_cost_pct),
            )
        except Exception as e:
            result = {"error": str(e)}

    context = {
        "result":            result,
        "form_data":         form_data,
        "drug_list":         ml_engine.DRUG_LIST,
        "drug_descriptions": ml_engine.DRUG_DESCRIPTIONS,
        "drug_options":      [(code, ml_engine.DRUG_DESCRIPTIONS.get(code, code)) for code in ml_engine.DRUG_LIST],
        "page":              "inventory",
    }
    return render(request, "forecasting/inventory.html", context)
