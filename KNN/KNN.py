# coding:utf-8

"""
@author : CLD
@time:2018/8/2820:01
@description: K-nearest neighbors
"""

from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inx,dataSet,labels,k):                                  #inx:输入值，dataSet：训练样本，labels：样本标签，
    dataSetSize=dataSet.shape[0]                                      #获得样本行数 shape[0]->行数 shape[1]->列数 .shape->维度
    diffMat=tile(inx,(dataSetSize,1))-dataSet                         #tile(A,B) 将A矩阵以B矩阵的方式铺开
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


if __name__=="__main__":
    group,labels=createDataSet();
    print(classify0([0,0],group,labels,3))