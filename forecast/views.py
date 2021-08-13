from django.shortcuts import render
from rest_framework.views import APIView

from illegal_fund_raising_forecast import settings
from .serializer import TestDatasetSerializer, TrainDatasetSerializer
from rest_framework.response import Response
from .models import TrainDataset
from utils import Res
import json
import random
import pandas as pd



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
                        data={"code": 200, "msg": "上传成功", "data": {
                            "test_id": obj.id,
                            "dataset_lines": random.randint(7731, 1000000),
                            "p_and_n_proportion": {
                                "positive": random.randint(1000, 10000),
                                "negative": random.randint(1000, 10000),
                            },
                            "area_distribution": [
                                {
                                    "name": '北京',
                                    "selected": True
                                },
                                {
                                    "name": '天津',
                                    'selected': True
                                },
                                {
                                    'name': '上海',
                                    'selected': True
                                },
                                {
                                    "name": '重庆',
                                    'selected': True
                                },
                                {
                                    'name': '河北',
                                    'selected': True
                                },
                                {
                                    'name': '河南',
                                    'selected': True
                                },
                                {
                                    'name': '四川',
                                    'selected': True
                                }
                            ]}}
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
                            "train_id": obj.id,
                            "dataset_lines": random.randint(7731, 1000000),
                            "p_and_n_proportion": {
                                "positive": random.randint(1000, 10000),
                                "negative": random.randint(1000, 10000),
                            },
                            "area_distribution": [
                                {
                                    "name": '北京',
                                    "selected": True
                                },
                                {
                                    "name": '天津',
                                    'selected': True
                                },
                                {
                                    'name': '上海',
                                    'selected': True
                                },
                                {
                                    "name": '重庆',
                                    'selected': True
                                },
                                {
                                    'name': '河北',
                                    'selected': True
                                },
                                {
                                    'name': '河南',
                                    'selected': True
                                },
                                {
                                    'name': '四川',
                                    'selected': True
                                }
                            ],
                            "url": obj.train.url,
                        }}
                        )


class TrainDownloadView(APIView):
    def get(self, request):
        train_id = request.query_params.get("train_id")
        if not train_id:
            return Response(json.loads(Res(code=400, msg="train_id is None", data=None).json()))
        try:
            obj = TrainDataset.objects.get(id=train_id)
            url = obj.train.url
        except Exception as e:
            url = None
        return Response(json.loads(Res(code=200, msg="success",
                                       data={
                                               "train_url": url
                                            }
                                       ).json()))


class TrainStartView(APIView):
    def get(self):
        pass


class ResultView(APIView):
    def get(self, request):
        pageIndex = int(request.query_params.get("pageIndex"))
        pageSize = int(request.query_params.get("pageSize"))
        df = pd.read_csv(settings.MEDIA_ROOT + "/testdata.csv")
        if pageIndex * pageSize > len(df) or pageIndex * pageSize <= 0:
            return Response({"code": 400, 'msg': "参数有误"})
        start_index = 0 if pageIndex * pageSize <= 0 else (pageIndex - 1) * pageSize
        df_ = df.iloc[start_index:pageIndex * pageSize, :]
        res = list(json.loads(df_.to_json(orient='index')).values())

        return Response(
            {
                "code": 200,
                "msg": "",
                'data': res,
                "pageTotal": len(df)
            }
        )




class ConsoleView(APIView):
    def get(self, request):
        return Response(
            {
                "code": 200,
                "msg": "success",
                "data": {
                    "running": 0,
                    "train_lines": 0,
                    "forecast_times": 0
                }

            }
        )


class CompanyView(APIView):
    def get(self, request):
        return Response(
            {
                "code": 200,
                "msg": "",
                "data":
                    [
                        {
                            "labels": ['合资', '独资', '国有', '私有', '集体所有制', '股份制', '有限责任制'],
                            "datasets": {
                                "data": [random.randint(100, 10000) for _ in range(7)]
                            },
                        },
                        {
                            "labels": ['地产', '银行', '互联网', '硬件', '半导体', '销售', '餐饮'],
                            "datasets": {
                                "data": [random.randint(100, 10000) for _ in range(7)]
                            },

                        },
                        {
                            "labels": ['采掘', '制造', '批发', '零售'],
                            "datasets": {
                                "data": [random.randint(100, 10000000) for _ in range(4)]
                            },
                        }
                    ]
            }
        )

class ScoreView(APIView):
    def get(self, request):
        return Response(
            {
                "code": 200,
                "msg": "",
                "data": {
                    "f1_score": random.random(),
                }
            }
        )
