"""
@Author: yanzx
@Date: 2021/8/8 16:30
@Description: 工具模块
"""


def my_jwt_response_payload_handler(token, user=None, request=None):
    return {
        'code': 200,
        'msg': '登录成功',
        "data":{
            'uid': user.id,
            'username': user.username,
            'token': token,
        }
    }
