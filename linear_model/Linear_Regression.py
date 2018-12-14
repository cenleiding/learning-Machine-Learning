# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/12/13
@description: 线性回归模型
"""
import numpy as np
from sklearn import datasets,linear_model,model_selection

# 糖尿病数据=》回归问题
def load_data_diabetes():
    diabetes = datasets.load_diabetes()
    # 1/4作为测试集
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def LinearRegression_test(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    print('predict:{}'.format(regr.predict(X_test)[:10]))
    print('y_test:{}'.format(y_test[:10]))
    print('Residual sum of squares:%.2f' % np.mean((regr.predict(X_test)-y_test)**2))
    print('Score: %.2f' % regr.score(X_test,y_test))

"""
简单使用LinearRegression准确率很低
"""
if __name__=='__main__':
    X_train,X_test,y_train,y_test = load_data_diabetes()
    LinearRegression_test(X_train,X_test,y_train,y_test)