# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/15
@description: 计算工具
"""
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(10,5),columns=['a','b','c','d','e'])
print(df)
print(df.pct_change(periods=1)) ## 计算值的百分比变化幅度
print(df.cov()) ## 计算属性之间的协方差
print(df.corr()) ## 计算属性之间的相关性
print(df.rank(1)) ## 值排名
print(df.count()) ## 计数
print(df.sum().sum()) ##求和
print(df.mean()) ## 平均值
print(df.median()) ## 中位数
print(df.min()) ## 最小值
print(df.max()) ## 最大值
print(df.std()) ## 标准差
print(df.var()) ## 方差

## 窗口操作 rolling() expanding() 限定范围，算术操作不变
print(df.rolling(window=2).mean())

