from django.urls import path
from . import views

app_name = "forecasting"

urlpatterns = [
    path("",                  views.dashboard,  name="dashboard"),
    path("results/",          views.results,    name="results"),
    path("forecast/",         views.forecast,   name="forecast"),
    path("drift/",            views.drift,      name="drift"),
    path("sample-csv/",       views.sample_csv, name="sample_csv"),
]
