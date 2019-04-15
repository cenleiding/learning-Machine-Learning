# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/15
@description: 处理分类型数据
"""
import numpy as np
import pandas as pd

df = pd.DataFrame({'A':list('abca'),'B':list('bccd')})
df['B'] = df.loc[:,'B'].astype('category')

cat_type = pd.api.types.CategoricalDtype(categories=["b","a"],ordered=True) # 自定义类别，和类别关系b>a
df['A'] = df['A'].astype(cat_type)
print(df['A'].dtypes)
