# coding:utf-8

"""
@author : CLD
@time:2018/9/520:24
@description: 决策树<ID3算法>
"""

from math import log
import operator

from DecisionTree import treePlotter


def calcShannonEnt(dataSet):                           #计算香农信息熵，默认数据集的最后一个列是类别标签
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob =float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):              #划分数据集
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] ==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  #注意extend和append的区别!
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):                  #选择最佳数据集划分
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[x[i] for x in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):                        #如果用完所有特征仍未能得出明确的分类，则通过投票获得最可能的分类
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[x[-1] for x in dataSet]
    if classList.count(classList[0])==len(classList): #如果已经相同则直接退出
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)                 #特征值用完，通过投票获得分类
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])                              #注意del() .remove .pop 的区别
    featValues=[x[bestFeat] for x in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):            #运行训练后的决策树
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    classLabel='未知'
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

def createDataSet():                                #测试数据
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,1,'no'],
             [1,1,'no'],
             [1,1,'no']]
    labels=['no surfacing','flippers']             #labels单纯为了好看
    return dataSet,labels

def storeTree(inputTree,filename):
    import pickle
    fw =open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

if __name__=='__main__':
    myDat,labels=createDataSet()
    myTree=createTree(myDat,labels.copy())
    treePlotter.createPlot(myTree)
    print(classify(myTree,labels,[1,1]))
    storeTree(myTree,'myTree.txt')
    print(grabTree('myTree.txt'))