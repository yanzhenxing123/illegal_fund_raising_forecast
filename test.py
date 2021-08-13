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

df  =pd.read_csv("./testdata.csv")

df = df.iloc[1:20, :]

print(type(df.to_json()))

