from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from .serializer import TestDatasetSerializer
from rest_framework.response import Response


class TestUploadView(APIView):
    def post(self, request):
        base_info = request.FILES.get("base_info")
        data = {
            "base_info": base_info,
            "user_id": request.user.id,
        }
        serializer = TestDatasetSerializer(data=data)
        if serializer.is_valid():
            serializer.create(data)
        return Response(status=200, data={"code":200})

class TrainUploadView(APIView):
    def post(self):
        pass


class TrainDownloadView(APIView):
    def get(self):
        pass


class TrainStartView(APIView):
    def get(self):
        pass


class ResultView(APIView):
    def get(self):
        pass
