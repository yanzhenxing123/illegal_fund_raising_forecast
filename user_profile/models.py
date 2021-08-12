from django.db import models
from django.contrib.auth.models import User


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    telephone = models.CharField(max_length=20, blank=True, unique=True, null=True)
    # 头像
    avatar = models.ImageField(upload_to="avatar/%Y%m%d", blank=True, null=True)

    # 个人简介
    bio = models.TextField(max_length=500, blank=True)

    def __str__(self):
        return 'user {}'.format(self.user.username)

    class Meta:
        db_table = "profile"
        # app_label = 'default'

class AuthUser(User):
    password = 1

