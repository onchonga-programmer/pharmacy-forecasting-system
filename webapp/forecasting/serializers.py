from rest_framework import serializers

from .models import ForecastResult, Medicine, SalesRecord


class MedicineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medicine
        fields = "__all__"


class SalesRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SalesRecord
        fields = "__all__"


class ForecastResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResult
        fields = "__all__"


class ForecastRequestSerializer(serializers.Serializer):
    drug_name = serializers.ChoiceField(
        choices=["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
    )
    forecast_date = serializers.DateField()


class InventoryRequestSerializer(serializers.Serializer):
    drug_name = serializers.ChoiceField(
        choices=["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
    )
    current_stock = serializers.FloatField()
    lead_time_weeks = serializers.IntegerField(min_value=0)
    service_level_pct = serializers.ChoiceField(choices=[90, 95, 98, 99])
    unit_cost = serializers.FloatField(min_value=0)
    order_cost = serializers.FloatField(min_value=0)
    holding_cost_pct = serializers.FloatField(min_value=0)
