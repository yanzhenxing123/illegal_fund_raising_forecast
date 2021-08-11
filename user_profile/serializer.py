"""
@Author: yanzx
@Date: 2020/11/30 21:49
@Description: 
"""
from datetime import datetime, timedelta
import re

from captcha.models import CaptchaStore
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



class UserRegSerializer(serializers.Serializer):
    username = serializers.CharField(min_length=1, max_length=150, required=True)
    password = serializers.CharField(min_length=6, max_length=20, required=True)
    email = serializers.EmailField(required=True)
    captcha = serializers.CharField(min_length=4, max_length=4, required=True,
                                    error_messages={
                                        "max_length": "图片验证码格式错误",
                                        "min_length": "图片验证码格式错误",
                                        "required": "请输入图片验证码"
                                    }, help_text="图片验证码")
    ima_id = serializers.CharField(required=True, write_only=True, allow_blank=False,help_text="图片验证码id")

    def validate_captcha(self, captcha):
        try:
            captcha = captcha.lower()
        except:
            raise serializers.ValidationError("图片验证码错误")
        image_code = CaptchaStore.objects.filter(
            id=self.initial_data['ima_id']).first()
        if image_code and datetime.now() > image_code.expiration:
            raise serializers.ValidationError('图片验证码过期')
        else:
            if image_code and image_code.response == captcha:
                pass
            else:
                raise serializers.ValidationError("图片验证码错误")

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