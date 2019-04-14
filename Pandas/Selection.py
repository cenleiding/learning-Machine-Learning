# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/14
@description:  数据的选择
"""
import numpy as np
import pandas as pd


dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.rand(8,4),index=dates,columns=['A','B','C','D'])

# 直接[列][行] 取值，属性取值
print(df['A'][dates[0]])
print(df[['A','B']])
# 行切片
print(df[0:4])
print(df['A'][0:4])
print(df['A'][0:4:2])

# 根据标签
# .loc[row_indexer,column_indexer]  输入为标签名
df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
print(df1.loc[:, :])
print(df1.loc[['a', 'b', 'c'], :])
print(df1.loc['d':, 'A':'C'])
print(df1.loc['a'] > 0)
print(df1.loc[:, df1.loc['a'] > 0])
print(df1.loc[lambda df2: df2.A > 0, :])

# 根据位置
# .iloc[row_indexer,column_indexer] 输入为数值坐标
print(df1.iloc[:3])
print(df1.iloc[:3, 1:3])
print(df1.iloc[[1, 3], [1, 2]])
print(df1.iloc[:, lambda df:[0, 1]])

# 随机选取
# .sample() 默认采样行
df2 = pd.DataFrame({'col1': [9, 8, 7, 6],
                    'weight_column': [0.6, 0.4, 0.1, 0]})
print(df2.sample(n=2))
print(df2.sample(n=6,replace=True))
print(df2.sample(frac=0.5))
print(df2.sample(n=6,weights='weight_column',replace=True)) #将某一列作为权重

# 获取标量值
# .at()标签名  .iat()坐标值
print(df2.at[0,'col1'])
print(df2.iat[1,1])

# 条件选择
# boolean
print(df2[df2>1])
# isin()
print(df2.isin([9,0.1]))
# where() 对不符合的还能操作
print(df2.where(df2>1,df2+100))
# mark() 和 where 相似
print(df2.mask(df2>1))
# query 贼灵活
print(df2.query('col1<weight_column'))
