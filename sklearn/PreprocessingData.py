# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/16
@description: 数据预处理
"""
from sklearn import preprocessing
import numpy as np

## 标准化

## .scale() 均值为0，方差为1。
## 以列为单位处理
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.],
                    [0., 1., 3]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))