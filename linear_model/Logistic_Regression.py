# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/12/13
@description: Logistic回归
"""
import numpy as np
import matplotlib .pyplot as plt
from sklearn import datasets,linear_model,model_selection

# 花数据=》分类问题
def load_data_iris():
    iris = datasets.load_iris()
    # 采用分层采样
    return model_selection.train_test_split(iris.data,iris.target,test_size=0.25,random_state=0,stratify=iris.target)

def LogisticRegression_test(*data):
    X_train,X_text,y_train,y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train,y_train)
    print('score:%.2f' % regr.score(X_text,y_test))

## multi_class:多分类问题策略
## --'ovr',one-vs-rest,默认
## --'multinomial',
def LogisticRegression_multinomial(*data):
    X_train,X_text,y_train,y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train,y_train)
    print('score:%.2f' % regr.score(X_text,y_test))

## C:正则项系数倒数，默认为1
def LogisticRegression_C(*data):
    X_train,X_text,y_train,y_test = data
    Cs = np.logspace(-2,4,num=100)
    scores = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_text,y_test))
    # plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel("C")
    ax.set_xscale('log')
    ax.set_ylabel("score")
    plt.show()

if __name__=='__main__':
    X_train,X_text,y_train,y_test = load_data_iris()
    print("一般调用")
    LogisticRegression_test(X_train,X_text,y_train,y_test)
    print("使用 multinomial")
    LogisticRegression_multinomial(X_train,X_text,y_train,y_test)
    print("调整正则参数C")
    LogisticRegression_C(X_train,X_text,y_train,y_test)


