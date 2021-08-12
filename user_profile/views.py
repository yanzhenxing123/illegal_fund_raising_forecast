import base64
import json

from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse
from rest_framework import mixins, viewsets, authentication, status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework_jwt.serializers import jwt_payload_handler
from rest_framework_jwt.settings import api_settings
from rest_framework_jwt.utils import jwt_encode_handler
from rest_framework_jwt.views import JSONWebTokenAPIView

from user_profile.my_auth import MyPermissions
from user_profile.serializer import UserRegSerializer, MyloginSerializer

from captcha.views import CaptchaStore, captcha_image

from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from datetime import datetime

from utils import Res

jwt_response_payload_handler = api_settings.JWT_RESPONSE_PAYLOAD_HANDLER


# 重写注册view
class MyJSONWebToken(JSONWebTokenAPIView):
    """"
    重写jwt的登录验证，含图片验证码
    """
    serializer_class = MyloginSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            user = serializer.object.get('user') or request.user
            token = serializer.object.get('token')
            response_data = jwt_response_payload_handler(token, user, request)
            response = Response(response_data)
            return response

        return Response(Res(200, str(serializer.errors), None).json(), status=status.HTTP_200_OK)


class ImageView(APIView):
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        hashkey = CaptchaStore.generate_key()
        try:
            # 获取图片id
            id_ = CaptchaStore.objects.filter(hashkey=hashkey).first().id
            imgage = captcha_image(request, hashkey)
            # 将图片转换为base64
            image_base = base64.b64encode(imgage.content)
            json_data = json.dumps({"id": id_, "image_base": image_base.decode('utf-8')})
        except:
            json_data = None
        return HttpResponse(json_data, content_type="application/json")


class RegisterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = UserRegSerializer(data=request.data)
        is_valid = serializer.is_valid()
        if not is_valid:
            return Response(status=200, data={"code": "400", "msg": serializer.errors, "data": None})

        uid = serializer.create(validated_data=request.data)
        return Response(status=201, data={"code": 200, "data": {
            "uid": uid
        }, "msg": "注册成功"})


class UserRegisterViewset(mixins.CreateModelMixin, mixins.UpdateModelMixin,
                          mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = UserRegSerializer
    queryset = User.objects.all()
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = self.perform_create(serializer)
        re_dict = serializer.data
        payload = jwt_payload_handler(user)
        re_dict['token'] = jwt_encode_handler(payload)
        headers = self.get_success_headers(serializer.data)
        return Response(re_dict, status=status.HTTP_201_CREATED, headers=headers)

    def get_serializer_class(self):
        '''
        重载GenericAPIView中的get_serializer_class函数，调用不同的序列化类，如果是create,
        就调用UserRegSerializer序列化，否则UserDetailSerializer序列化
        :return:
        '''
        return UserRegSerializer

    def get_permissions(self):
        '''
        重载APIview中的get_perimissions函数，如果是新增用户则不用登录，否则必须登录
        :return:
        '''
        if self.action == 'retrieve':
            return [permissions.IsAuthenticated()]
        elif self.action == 'create':
            return []
        return []

    def get_object(self):
        '''
        返回当前用户
        :return:
        '''
        return self.request.user

    def perform_create(self, serializer):
        return serializer.save()


class RegisterView2(APIView):
    permission_classes = []

    def post(self, request):
        serializer = UserRegSerializer(data=request.data)
        is_valid = serializer.is_valid(raise_exception=True)
        if is_valid:
            re_dict = serializer.data
            return Response(re_dict, status=status.HTTP_201_CREATED)
        return Response(serializer.errors)


class TestView(APIView):
    def get(self, request):
        return Response(data={"dada"})
