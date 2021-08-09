"""
@Author: yanzx
@Date: 2020/11/30 21:49
@Description: 
"""
import re

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import TestDataset


class TestDatasetSerializer(serializers.Serializer):
    base_info = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    user_id = serializers.IntegerField()

    def create(self, validated_data):
        print(validated_data['base_info'])
        TestDataset.objects.create(user_id=validated_data['user_id'], base_info=validated_data['base_info'])
