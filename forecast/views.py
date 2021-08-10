from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from .serializer import TestDatasetSerializer, TrainDatasetSerializer
from rest_framework.response import Response


class TestUploadView(APIView):
    def post(self, request):
        base_info = request.FILES.get("base_info")
        annual_report_info = request.FILES.get("annual_report_info")
        tax_info = request.FILES.get("tax_info")
        change_info = request.FILES.get("change_info")
        news_info = request.FILES.get("news_info")
        other_info = request.FILES.get("other_info")
        data = {
            "base_info": base_info,
            "annual_report_info": annual_report_info,
            "tax_info": tax_info,
            "change_info": change_info,
            "news_info": news_info,
            "other_info": other_info,
            "user_id": request.user.id,
        }
        serializer = TestDatasetSerializer(data=data)
        obj = serializer.create(data)
        return Response(status=200,
                        data={"code": 200, "msg":"上传成功", "data":{
                            "test_id": obj.id,
                            "dataset_lines": "还未实现",
                            "p_and_n_proportion": "还未实现",
                            "area_distribution": "还未实现",
                        }}
        )


class TrainUploadView(APIView):
    def post(self, request):
        train = request.FILES.get("train")
        data = {
            "train": train,
            "user_id": request.user.id,
        }
        serializer = TrainDatasetSerializer(data=data)
        obj = serializer.create(validated_data=data)
        return Response(status=200,
                        data={"code": 200, "msg": "上传成功", "data": {
                            "test_id": obj.id,
                            "dataset_lines": "还未实现",
                            "p_and_n_proportion": "还未实现",
                            "area_distribution": "还未实现",
                            "url": obj.train.url,
                        }}
                        )

class TrainDownloadView(APIView):
    def get(self):
        pass


class TrainStartView(APIView):
    def get(self):
        pass


class ResultView(APIView):
    def get(self):
        pass
