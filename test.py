"""
@Author: yanzx
@Date: 2021-08-10 09:27:55
@Desc: 
"""

import time

li = [str(i) + "闫振兴" for i in range(1000000)]

li_s = set(li)

start_time1 = time.time()

if "100000闫振兴" in li:
    print(time.time() - start_time1)

start_time2 = time.time()

if "100000闫振兴" in li_s:
    print(time.time() - start_time2)



import pandas as pd
import numpy as np
import json

df  =pd.read_csv("./testdata.csv")

df = df.iloc[1:20, :]
res = list(json.loads(df.to_json(orient='index')).values())
print(res)
data_array = np.array(df)
# 然后转化为list形式
data_list =data_array.tolist()

# print(data_list)



