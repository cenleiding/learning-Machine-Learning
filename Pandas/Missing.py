# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/15
@description: 处理丢失数据
"""
import numpy as np
import pandas as pd

## 注意用 isna() notna()去判断是否为NAN，因为 np.nan!=np.nan
## 运算时会无视NaN值

## fillna() 填充
s = pd.DataFrame({'a':[1,2,np.nan],
                  'b':[0,0,np.nan],
                  'c':[2,6,np.nan]})

s['a'].fillna(88,inplace=True)  ## 注意如果不写inplace则不会对原数组造成影响。

s['b'].fillna(method='pad',inplace=True) ## method= pad/ffill 用前面的值填充。 bfill/backfill 用后面的值填充

s['c'].fillna(s['c'].mean(),inplace=True)

## dropna() 抛弃
s = pd.DataFrame({'a':[1,2,np.nan],
                  'b':[0,0,np.nan],
                  'c':[2,6,np.nan]})

s.dropna(axis=0) # 抛弃行
s.dropna(axis=1) # 抛弃列

## interpolation 插值，很6，多种模式,需要scipy支持
df = pd.DataFrame({'A':[1,2.1,np.nan,4.7,5.6,6.8],
                   'B':[0.25,np.nan,np.nan,4,12.2,14.4]})

print(df.interpolate())               # linear
print(df.interpolate(method='pchip')) # 累计分布
print(df.interpolate(method='akima')) # 平滑绘图

## replace() 值的简单替换
df = pd.DataFrame({'a':[0,1,2,3,4],
                   'b':[5,6,7,8,9]})
df.replace([0,1],[10,11],inplace=True)
df.replace({3:30,4:40},inplace=True)
df.replace({'a':0,'b':5},100,inplace=True)
print(df)
