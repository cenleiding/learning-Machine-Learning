# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/16
@description: 跑着玩,oneHot,sparse
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import numpy as np
import scipy.sparse as sparse


iris = datasets.load_iris()
data = iris.data

## 假装onehot
# oht = OneHotEncoder(categories='auto')
# print(np.shape(data[:,1:]))
# print(np.shape(oht.fit_transform(data[:,0].reshape(-1,1))))
# data = np.hstack((data[:,1:],oht.fit_transform(data[:,0].reshape(-1,1)).toarray()))
# print(np.shape(data))

target = iris.target
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2)


## 假装sparse
# X_train = sparse.hstack((X_train[:,1:].reshape(-1,3),oht.fit_transform(X_train[:,0].reshape(-1,1))))
# X_test = sparse.hstack((X_test[:,1:].reshape(-1,3),oht.fit_transform(X_test[:,0].reshape(-1,1))))


## 数据
lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)

## 属性
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': ['l2','AUC'],  # 评估函数
    'num_leaves': 31,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

##
gbm = lgb.train(params=params,
                train_set=lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=20,
                early_stopping_rounds=5)

gbm.save_model('./model.txt')

gbm = lgb.Booster(model_file='./model.txt')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(y_pred)
print(y_test)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


