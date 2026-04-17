from django.urls import path

from . import views

app_name = "forecasting"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("results/", views.results, name="results"),
    path("forecast/", views.forecast, name="forecast"),
    path("drift/", views.drift, name="drift"),
    path("inventory/", views.inventory, name="inventory"),
    path("sample-csv/", views.sample_csv, name="sample_csv"),
    path("api/", views.api_root, name="api_root"),
    path("api/drugs/", views.drug_list, name="drug_list"),
    path("api/inventory/", views.inventory_list, name="inventory_list"),
    path("api/inventory/<str:drug_code>/", views.inventory_detail, name="inventory_detail"),
    path("api/forecast/<str:drug_name>/<str:forecast_date>/", views.forecast_api, name="api_forecast"),
    path(
        "api/forecast/<str:drug_name>/<str:forecast_date>/chart/",
        views.forecast_chart_api,
        name="api_forecast_chart",
    ),
    path("api/inventory/recommend/", views.inventory_recommendation, name="inventory_recommendation"),
    path("api/forecast/batch/", views.batch_forecast, name="batch_forecast"),
    path("api/sales/", views.sales_log, name="sales_log"),
]
