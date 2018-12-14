# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/12/13
@description: 用最基础的写法实现logistic回归
"""
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('source/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights+alpha*dataMatrix.T*error
    return weights

## 随机梯度上升
def stocGradAscent0(dataMatIn,classLabels,numIter = 150):
    dataMatrix = np.array(dataMatIn)
    classLabels = np.array(classLabels).T
    m,n = np.shape(dataMatrix)
    weights = np.ones(n,dtype=np.float32)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(1+j+i)+0.01
            randIndex = int(np.random.uniform(0,m))  ## 随机采样
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] -h
            a = alpha*error*dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[randIndex]
            np.delete(dataMatrix,randIndex,axis=0)
    return weights

def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    flg = plt.figure()
    ax = flg.add_subplot(1,1,1)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3,3,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y.T)
    plt.show()



if __name__=='__main__':
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent0(dataArr,labelMat)
    plotBestFit(weights)
