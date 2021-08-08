from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from user_profile.my_auth import MyPermissions
from user_profile.serializer import UserRegisterSerializer


class RegisterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        is_valid = serializer.is_valid()
        if not is_valid:
            return Response(status=400, data=serializer.errors)
        uid = serializer.create(serializer.data)
        return Response(status=201, data={"code": 200, "uid": uid, "msg": "注册成功"})


class TestView(APIView):
    def get(self, request):
        return Response(data={"dada"})
