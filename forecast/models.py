from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class TestDataset(models.Model):
    # 一个用户对应多个数据集
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    base_info = models.FileField(upload_to="train/%Y%m%d/", blank=True, null=True)

    class Meta:
        db_table = 'test'
