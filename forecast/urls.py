from django.urls import path
from . import views

app_name = 'forecast'

urlpatterns = [
    path('test/upload/', views.TestUploadView.as_view(), name="test upload"),
    path('train/upload/', views.TrainUploadView.as_view(), name="train_upload"),
    path('train/download/', views.TrainDownloadView.as_view(), name="train_download"),
    path('result/', views.ResultView.as_view(), name="result"),
    path('train/start/', views.TrainStartView.as_view(), name="train_start")
]
