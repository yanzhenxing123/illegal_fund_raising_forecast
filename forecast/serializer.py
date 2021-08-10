"""
@Author: yanzx
@Date: 2020/11/30 21:49
@Description: 
"""
from rest_framework import serializers
from .models import TestDataset, TrainDataset


class TestDatasetSerializer(serializers.Serializer):
    base_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    annual_report_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    tax_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    change_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    news_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    other_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)

    user_id = serializers.IntegerField()

    def create(self, validated_data):
        return TestDataset.objects.create(
            user_id=validated_data['user_id'],
            base_info=validated_data['base_info'],
            annual_report_info=validated_data['annual_report_info'],
            tax_info=validated_data['tax_info'],
            change_info=validated_data['change_info'],
            news_info=validated_data['news_info'],
            other_info=validated_data['other_info'],
        )


class TrainDatasetSerializer(serializers.Serializer):
    train_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)

    user_id = serializers.IntegerField()

    def create(self, validated_data):
        return TrainDataset.objects.create(
            user_id=validated_data['user_id'],
            train=validated_data['train'],
        )
