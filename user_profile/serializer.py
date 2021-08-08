"""
@Author: yanzx
@Date: 2020/11/30 21:49
@Description: 
"""
import re

from rest_framework import serializers
from django.contrib.auth.models import User

from .models import Profile


def validate_username(username):
    regex = r'^[\w.@+-]+$'
    res = re.match(regex, username)
    if not res:
        message = "Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only."
        raise serializers.ValidationError(message)


class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField(min_length=1, max_length=150, required=True)
    password = serializers.CharField(min_length=6, max_length=20, required=True)

    def validate_username(self, username):
        print(username)
        validate_username(username)
        return username


class UserRegisterSerializer(serializers.Serializer):
    username = serializers.CharField(min_length=1, max_length=150, required=True)
    password = serializers.CharField(min_length=6, max_length=20, required=True)
    email = serializers.EmailField(required=True)

    def validate_username(self, username):
        validate_username(username)
        is_exist = User.objects.filter(username=username)

        if is_exist:
            raise serializers.ValidationError("用户名已存在")
        return username

    def validate_telephone(self, telephone):
        is_exist = Profile.objects.filter(telephone=telephone)
        if is_exist:
            raise serializers.ValidationError("手机号已存在")
        return telephone

    def create(self, validated_data):
        user = User.objects.create_user(username=validated_data['username'], password=validated_data['password'], email=validated_data['email'])
        Profile.objects.create(user=user)
        return user.id