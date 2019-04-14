# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/14
@description: Pandas.IO
"""
import numpy as np
import pandas as pd
from pandas.compat import StringIO, BytesIO

## 常用 CSV,JOSN
## read_csv()  to_csv()
## read_josn() to_json()

data = ('col1,col2,col3\n'
        'a,b,1\n'
        'a,b,2\n'
        'c,d,3')

dataFrame = pd.read_csv(StringIO(data),sep=',',encoding='utf-8')
dataFrame.to_csv("./text.csv",sep=",",encoding="utf-8",index=False)
dataFrame = pd.read_csv("./text.csv")

# usecols 选择列
dataFrame = pd.read_csv(StringIO(data),usecols=['col1','col3'])
dataFrame = pd.read_csv(StringIO(data),usecols=[0,2])
dataFrame = pd.read_csv(StringIO(data),usecols=lambda x:x.upper() in ['COL1','COL3'])

# skiprows 选择移除行
dataFrame = pd.read_csv(StringIO(data),skiprows=[2])
print(dataFrame)
dataFrame = pd.read_csv(StringIO(data),skiprows=lambda x:x%2!=0)


# dtype 指定每行的类型
dataFrame = pd.read_csv(StringIO(data),dtype={'col1':object,'col2':object,'col3':np.int64})

# names header 替换列名
dataFrame = pd.read_csv(StringIO(data), names=['foo', 'bar', 'baz'], header=0)
dataFrame = pd.read_csv(StringIO(data), names=['foo', 'bar', 'baz'], header=None)

# print(dataFrame)