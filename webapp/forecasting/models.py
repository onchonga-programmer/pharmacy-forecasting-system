from django.db import models


class Medicine(models.Model):
	class VolumeSegment(models.TextChoices):
		HIGH = "High", "High"
		MEDIUM = "Medium", "Medium"
		LOW = "Low", "Low"

	drug_code = models.CharField(max_length=10, unique=True)
	description = models.CharField(max_length=255)
	volume_segment = models.CharField(max_length=10, choices=VolumeSegment.choices)
	current_stock = models.FloatField(default=0)
	unit_cost = models.DecimalField(max_digits=10, decimal_places=2)
	lead_time_weeks = models.IntegerField(default=2)
	reorder_level = models.FloatField(default=0)

	def __str__(self):
		return f"{self.drug_code} - {self.description}"


class SalesRecord(models.Model):
	medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE)
	quantity_sold = models.FloatField()
	date = models.DateField()
	recorded_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"{self.medicine.drug_code} on {self.date}"


class ForecastResult(models.Model):
	class ModelUsed(models.TextChoices):
		XGBOOST = "XGBoost", "XGBoost"
		HOLT_WINTERS = "Holt-Winters", "Holt-Winters"

	medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE)
	forecast_date = models.DateField()
	prediction = models.FloatField()
	lower_bound = models.FloatField()
	upper_bound = models.FloatField()
	model_used = models.CharField(max_length=20, choices=ModelUsed.choices)
	confidence_tier = models.CharField(max_length=10)
	created_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"{self.medicine.drug_code} forecast for {self.forecast_date}"


