# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/16
@description: 处理丢失值
"""
import numpy as np

## SimpleImputer  选择不同的策略插值（）
# 'mean':均值 =》数值
# 'median':中位数 =》数值
# 'most_frequent':高频词 =》数值、字符串
# 'constant':常量 =》数值、字符串 fill_value
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
print(imp.fit_transform([[1, 2], [np.nan, 3], [7, 6]]))

