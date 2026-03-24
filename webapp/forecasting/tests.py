

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

    def test_inventory_get_returns_200(self):
        response = self.client.get(reverse("forecasting:inventory"))
        self.assertEqual(response.status_code, 200)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inventory Optimisation
# ─────────────────────────────────────────────────────────────────────────────

class InventoryOptimisationTest(TestCase):
    """ml_engine.compute_inventory_recommendation() should return valid metrics."""

    DRUG    = "M01AB"
    PARAMS  = dict(
        current_stock     = 500.0,
        lead_time_weeks   = 2,
        service_level_pct = 95.0,
        unit_cost         = 10.0,
        order_cost        = 50.0,
        holding_cost_pct  = 25.0,
    )

    def setUp(self):
        self.result = ml_engine.compute_inventory_recommendation(
            drug_name=self.DRUG, **self.PARAMS
        )

    def test_no_error_returned(self):
        self.assertIsNone(
            self.result.get("error"),
            f"compute_inventory_recommendation returned error: {self.result.get('error')}",
        )

    def test_required_keys_present(self):
        required = {
            "safety_stock", "reorder_point", "eoq", "weeks_of_coverage",
            "status", "suggested_order", "avg_weekly_demand", "std_weekly_demand",
            "annual_demand", "order_value", "stock_pct", "rop_pct", "ss_pct",
        }
        missing = required - set(self.result.keys())
        self.assertSetEqual(missing, set(), f"Result missing keys: {missing}")

    def test_safety_stock_is_non_negative(self):
        self.assertGreaterEqual(self.result["safety_stock"], 0)

    def test_reorder_point_greater_than_safety_stock(self):
        self.assertGreater(
            self.result["reorder_point"],
            self.result["safety_stock"],
            "Reorder point must exceed safety stock",
        )

    def test_eoq_is_positive(self):
        self.assertIsNotNone(self.result["eoq"])
        self.assertGreater(self.result["eoq"], 0)

    def test_weeks_of_coverage_positive(self):
        self.assertGreater(self.result["weeks_of_coverage"], 0)

    def test_suggested_order_positive(self):
        self.assertGreater(self.result["suggested_order"], 0)

    def test_status_is_valid_value(self):
        valid_statuses = {"CRITICAL", "ORDER NOW", "APPROACHING", "ADEQUATE"}
        self.assertIn(self.result["status"], valid_statuses)

    def test_critical_status_when_stock_zero(self):
        r = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=0.0, lead_time_weeks=2,
            service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        self.assertEqual(r["status"], "CRITICAL")

    def test_adequate_status_when_stock_very_high(self):
        r = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=99999.0, lead_time_weeks=2,
            service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        self.assertEqual(r["status"], "ADEQUATE")

    def test_higher_service_level_gives_larger_safety_stock(self):
        r95 = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=500.0, lead_time_weeks=2,
            service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        r99 = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=500.0, lead_time_weeks=2,
            service_level_pct=99.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        self.assertGreater(
            r99["safety_stock"], r95["safety_stock"],
            "99% service level should require more safety stock than 95%",
        )

    def test_longer_lead_time_gives_larger_reorder_point(self):
        r2 = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=500.0, lead_time_weeks=2,
            service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        r4 = ml_engine.compute_inventory_recommendation(
            drug_name="M01AB", current_stock=500.0, lead_time_weeks=4,
            service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
            holding_cost_pct=25.0,
        )
        self.assertGreater(
            r4["reorder_point"], r2["reorder_point"],
            "Longer lead time should produce a higher reorder point",
        )

    def test_all_drugs_compute_without_error(self):
        for drug in ml_engine.DRUG_LIST:
            r = ml_engine.compute_inventory_recommendation(
                drug_name=drug, current_stock=500.0, lead_time_weeks=2,
                service_level_pct=95.0, unit_cost=10.0, order_cost=50.0,
                holding_cost_pct=25.0,
            )
            self.assertIsNone(
                r.get("error"),
                f"compute_inventory_recommendation failed for {drug}: {r.get('error')}",
            )

    def test_inventory_post_returns_200(self):
        client = self.client
        response = client.post(
            reverse("forecasting:inventory"),
            data={
                "drug_name":        "M01AB",
                "current_stock":    "500",
                "lead_time_weeks":  "2",
                "service_level":    "95",
                "unit_cost":        "10.00",
                "order_cost":       "50.00",
                "holding_cost_pct": "25",
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_inventory_result_in_response_context(self):
        response = self.client.post(
            reverse("forecasting:inventory"),
            data={
                "drug_name":        "M01AB",
                "current_stock":    "500",
                "lead_time_weeks":  "2",
                "service_level":    "95",
                "unit_cost":        "10.00",
                "order_cost":       "50.00",
                "holding_cost_pct": "25",
            },
        )
        self.assertIn("result", response.context)
        self.assertIsNone(response.context["result"].get("error"))

