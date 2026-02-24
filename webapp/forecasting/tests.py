"""
Tests for the Pharmacy Forecasting System webapp.

Run with:
    python manage.py test forecasting
"""

from django.test import TestCase, Client
from django.urls import reverse

from . import ml_engine


# ─────────────────────────────────────────────────────────────────────────────
# 1. ML Engine — model loading
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoadTest(TestCase):
    """ml_engine.load_model() should load the saved XGBoost model without error."""

    def test_model_loads_without_error(self):
        model = ml_engine.load_model()
        self.assertIsNotNone(model, "load_model() returned None")

    def test_model_has_correct_feature_count(self):
        """Model must have been trained on exactly 47 clean features (no leakage)."""
        model = ml_engine.load_model()
        features = ml_engine._get_model_feature_names()
        self.assertEqual(
            len(features), 47,
            f"Expected 47 features after leakage removal, got {len(features)}",
        )

    def test_model_does_not_include_leaky_feature(self):
        """total_quantity_log (= log of target) must NOT be in the feature list."""
        features = ml_engine._get_model_feature_names()
        self.assertNotIn(
            "total_quantity_log", features,
            "Data leakage detected: total_quantity_log is still a model feature",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ML Engine — live forecast generation
# ─────────────────────────────────────────────────────────────────────────────

class ForecastGenerationTest(TestCase):
    """ml_engine.generate_forecast() should return a valid prediction dict."""

    DRUG        = "M01AB"
    FORECAST_DT = "2019-01-07"

    def setUp(self):
        self.result = ml_engine.generate_forecast(self.DRUG, self.FORECAST_DT)

    def test_no_error_returned(self):
        self.assertIsNone(
            self.result.get("error"),
            f"generate_forecast returned an error: {self.result.get('error')}",
        )

    def test_prediction_is_positive_number(self):
        pred = self.result.get("prediction")
        self.assertIsNotNone(pred, "prediction key missing from result")
        self.assertIsInstance(pred, float, "prediction should be a float")
        self.assertGreaterEqual(pred, 0.0, "prediction should be non-negative")

    def test_prediction_is_plausible(self):
        """Prediction should be within a reasonable sales range (0–2000 units/week)."""
        pred = self.result["prediction"]
        self.assertLess(pred, 2000, f"Prediction {pred} looks unrealistically high")

    def test_result_contains_required_keys(self):
        required = {
            "drug", "drug_description", "volume_segment",
            "date", "prediction", "lower_bound", "upper_bound",
            "seasonal_avg", "recent_avg", "mae_2018",
        }
        missing = required - set(self.result.keys())
        self.assertSetEqual(missing, set(), f"Result missing keys: {missing}")

    def test_confidence_bounds_are_ordered(self):
        lower = self.result["lower_bound"]
        pred  = self.result["prediction"]
        upper = self.result["upper_bound"]
        self.assertLessEqual(lower, pred,  "lower_bound should be ≤ prediction")
        self.assertLessEqual(pred,  upper, "prediction should be ≤ upper_bound")

    def test_drug_name_echoed_correctly(self):
        self.assertEqual(self.result["drug"], self.DRUG)

    def test_forecast_date_echoed_correctly(self):
        self.assertEqual(self.result["date"], "2019-01-07")

    def test_all_eight_drugs_forecast_without_error(self):
        """generate_forecast() should succeed for every drug in the dataset."""
        for drug in ml_engine.DRUG_LIST:
            result = ml_engine.generate_forecast(drug, self.FORECAST_DT)
            self.assertIsNone(
                result.get("error"),
                f"generate_forecast failed for {drug}: {result.get('error')}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Views — HTTP responses
# ─────────────────────────────────────────────────────────────────────────────

class DashboardViewTest(TestCase):
    """Dashboard page should return HTTP 200 and render key metrics."""

    def setUp(self):
        self.client = Client()

    def test_dashboard_returns_200(self):
        response = self.client.get(reverse("forecasting:dashboard"))
        self.assertEqual(
            response.status_code, 200,
            f"Dashboard returned {response.status_code}, expected 200",
        )

    def test_dashboard_contains_mae_value(self):
        """Page should display the 2018 MAE (11.45) from the saved metrics file."""
        response = self.client.get(reverse("forecasting:dashboard"))
        self.assertContains(response, "11.45")

    def test_dashboard_contains_drug_codes(self):
        """At least one drug ATC code should appear in the rendered page."""
        response = self.client.get(reverse("forecasting:dashboard"))
        self.assertContains(response, "N02BE")

    def test_results_page_returns_200(self):
        response = self.client.get(reverse("forecasting:results"))
        self.assertEqual(response.status_code, 200)

    def test_results_page_with_drug_param_returns_200(self):
        response = self.client.get(
            reverse("forecasting:results"), {"drug": "M01AB", "year": "2018"}
        )
        self.assertEqual(response.status_code, 200)

    def test_forecast_get_returns_200(self):
        response = self.client.get(reverse("forecasting:forecast"))
        self.assertEqual(response.status_code, 200)

    def test_drift_page_returns_200(self):
        response = self.client.get(reverse("forecasting:drift"))
        self.assertEqual(response.status_code, 200)
