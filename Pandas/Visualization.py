# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/15
@description: 可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ts = pd.Series(np.random.rand(1000),index=pd.date_range('1/1/2000',periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()