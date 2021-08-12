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
from rest_framework_jwt.serializers import JSONWebTokenSerializer

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

    def create(self, validated_data) -> int:
        user = User.objects.create_user(username=validated_data['username'], password=validated_data['password'], email=validated_data['email'])
        Profile.objects.create(user=user)
        return user.id
    
# 登录序列化

from rest_framework_jwt.settings import api_settings

jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
from django.contrib.auth import authenticate
from rest_framework_jwt.compat import PasswordField


class MyloginSerializer(JSONWebTokenSerializer):
    """
    从写登录序列化
    """
    def __init__(self, *args, **kwargs):
        """
        Dynamically add the USERNAME_FIELD to self.fields.
        """
        super(JSONWebTokenSerializer, self).__init__(*args, **kwargs)

        self.fields[self.username_field] = serializers.CharField()
        self.fields['password'] = PasswordField(write_only=True)
        self.fields['captcha'] = serializers.CharField(min_length=4, max_length=4, required=True,
                                                       error_messages={
                                                           "max_length": "图片验证码格式错误",
                                                           "min_length": "图片验证码格式错误",
                                                           "required": "请输入图片验证码"
                                                       })
        self.fields['ima_id'] = serializers.CharField(required=True, allow_blank=False)

    def validate_captcha(self, captcha):
        image_code = CaptchaStore.objects.filter(
            id=self.initial_data['ima_id']).first()
        five_minute_ago = datetime.now() - timedelta(hours=0, minutes=5, seconds=0)
        if image_code and five_minute_ago > image_code.expiration:
            raise serializers.ValidationError('验证码过期')
        else:
            if image_code and (image_code.response == captcha or image_code.challenge == captcha):
                pass
            else:
                raise serializers.ValidationError("图片验证码错误")

    def validate(self, attrs):
        del attrs["ima_id"]
        del attrs["captcha"]
        credentials = {
            self.username_field: attrs.get(self.username_field),
            'password': attrs.get('password')
        }

        if all(credentials.values()):
            user = authenticate(**credentials)

            if user:
                if not user.is_active:
                    msg = '用户未激活.'
                    raise serializers.ValidationError(msg)

                payload = jwt_payload_handler(user)

                return {
                    'token': jwt_encode_handler(payload),
                    'user': user
                }
            else:
                msg = '用户名或密码错误'
                raise serializers.ValidationError(msg)
        else:
            msg = '用户名和密码必须输入'
            msg = msg.format(username_field=self.username_field)
            raise serializers.ValidationError(msg)

