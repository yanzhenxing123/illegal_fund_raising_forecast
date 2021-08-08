"""
@Author: yanzx
@Date: 2021/8/8 15:21
@Description: 
"""

# 权限类
from rest_framework.permissions import BasePermission
from rest_framework_jwt.authentication import jwt_decode_handler, JSONWebTokenAuthentication


class MyPermissions(BasePermission):
    # message='你是2b'
    def has_permission(self, request, view):
        # 代表是超级用户
        if request.user:
            # 如何去除type对应的文字  get_字段名_display()
            return True
        return False

