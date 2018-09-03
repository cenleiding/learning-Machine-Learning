# coding:utf-8

"""
@author : CLD
@time:2018/8/2820:01
@description: K-nearest neighbors
             ~约会网站~
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inx,dataSet,labels,k):                                  #分类器  inx:输入值，dataSet：训练样本，labels：样本标签，
    dataSetSize=dataSet.shape[0]                                      #获得样本行数 shape[0]->行数 shape[1]->列数 .shape->维度
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet                         #tile(A,B) 将A矩阵以B矩阵的方式铺开
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)                                 #axis=None 所有值相加 axis=1 行相加 axis=0 列相加
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()                            #argsort() 获得降序排列的索引值
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #operator.itemgetter()获得指定域的值(1)->第一列的值，(1,0)->第一列第0行的值，也可以用lambda x:x[1]来代替
    return sortedClassCount[0][0]

def file2matrix(filename):                                #文件读取
    fr=open(filename,'r')
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))                 #注意numpy的array和自带的array是不一样的。。不要混淆！
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()                                  #去除头尾的空格和换行符
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):                                     #归一化程序
    minValue=dataSet.min(0)                                #min(0),使其获得列最小值
    maxValue=dataSet.max(0)
    ranges=maxValue-minValue
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minValue,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return  normDataSet,ranges,minValue

def datingClassTest():                                           #错误率测试器，hoRatio：设置10%的数据为测试样本，90%为训练样本
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minvalue=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("预测值：%d. 真实值：%d" % (classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):errorCount+=1.0
    print("错误率：%f" % (errorCount/numTestVecs))

def classifyPerson():                                                          #模拟测试！
    resultList=['bad','so','good']
    percent=float(input("游戏时间："))
    ffMiles=float(input("每年旅游距离："))
    iceCream=float(input("每年冰淇淋量："))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minvalue=autoNorm(datingDataMat)
    inArr=np.array([ffMiles,percent,iceCream])
    classifierResult=classify0((inArr-minvalue)/ranges,normMat,datingLabels,3)
    print('牵手可能性：'+resultList[classifierResult])

if __name__=='__main__':
    dataMat,dataLabels=file2matrix('datingTestSet2.txt')
    fig=matplotlib.pyplot.figure()
    ax=fig.add_subplot(111)
    ax.scatter(dataMat[:,1],dataMat[:,2], 15 * np.array(dataLabels))
    plt.show()
    datingClassTest()
    classifyPerson()
