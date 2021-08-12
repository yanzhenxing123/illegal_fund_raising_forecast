"""
@Author: yanzx
@Date: 2021-08-10 09:27:55
@Desc: 
"""

import base64

with open("demo.png", "rb") as f:
    content = f.read()

res = base64.b64encode(content)
print(type(res.decode("utf8")))
