from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class TestDataset(models.Model):
    # 一个用户对应多个数据集
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    base_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)
    annual_report_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)
    tax_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)
    change_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)
    news_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)
    other_info = models.FileField(upload_to="test/%Y%m%d/", blank=True, null=True)

    class Meta:
        db_table = 'test'


class TrainDataset(models.Model):
    # 一个用户对应多个数据集
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    train = models.FileField(upload_to="train/%Y%m%d/", blank=True, null=True)

    class Meta:
        db_table = 'train'

