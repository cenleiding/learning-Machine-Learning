# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/12/13
@description: Ridge回归，即加入了L2范数惩罚。
"""
import numpy as np
import matplotlib .pyplot as plt
from sklearn import datasets,linear_model,model_selection

# 糖尿病数据=》回归问题
def load_data_diabetes():
    diabetes = datasets.load_diabetes()
    # 1/4作为测试集
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def Ridge_alpha_test(*data):
    X_train,X_test,y_train,y_test = data
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores = []
    for i,alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha = alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    ## plt
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale('log')
    ax.set_ylabel("score")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test = load_data_diabetes()
    Ridge_alpha_test(X_train,X_test,y_train,y_test)
