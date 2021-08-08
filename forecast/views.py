from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView


class TestUploadView(APIView):
    def post(self):
        pass


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
