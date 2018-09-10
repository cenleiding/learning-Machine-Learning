# coding:utf-8

"""
@author : CLD
@time:2018/9/1021:11
@description:
              ML习题：4.3
              存在数值特征
"""

from math import log

from DecisionTree import treePlotter


def file2matrix(filename):
    data = []
    with open(filename,'r',encoding='UTF-8') as fr:
        for line in fr.readlines():
            list=[]
            for d in line.strip().split(','):
                list.append(d)
            data.append(list)
    return data

def calcShannonEnt(dataSet):
    num=len(dataSet)
    shannonEnt = 0.0
    labelCount={}
    for label in dataSet:
        labelCount[label[-1]] = labelCount[label[-1]]+1 if label[-1] in labelCount.keys() else 1
    for value in labelCount.values():
        prob=float(value)/num
        shannonEnt -= prob*log(prob+0.000001,2)
    return shannonEnt

def majorityCnt(classList):
    labelCount = {}
    for label in classList:
        labelCount[label[-1]] = labelCount[label[-1]] + 1 if label[-1] in labelCount.keys() else 1
    sortedClassCount=sorted(labelCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] ==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def splitDataSet_num(dataSet,axis,value):
    rightDataSet=[]
    leftDateSet=[]
    for featVec in dataSet:
        reducedFeatVec = featVec[:axis]
        reducedFeatVec.extend(featVec[axis + 1:])
        if featVec[axis] >=value:
            rightDataSet.append(reducedFeatVec)
        else:
            leftDateSet.append(reducedFeatVec)
    return rightDataSet,leftDateSet

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestValue=0.0
    for i in range(numFeatures):
        featList = [x[i] for x in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        if(is_number(featList[0])):
            for value in uniqueVals:
                newEntropy = 0.0
                rightDataSet, leftDateSet=splitDataSet_num(dataSet,i,value)
                probr = len(rightDataSet) / float(len(dataSet))
                probl = len(leftDateSet) / float(len(dataSet))
                newEntropy += probr * calcShannonEnt(rightDataSet)
                newEntropy += probl * calcShannonEnt(leftDateSet)
                infoGain = baseEntropy - newEntropy
                if infoGain > bestInfoGain:
                    bestInfoGain = infoGain
                    bestFeature = i
                    bestValue=value
        else:
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature,bestValue



def createTree(dataSet,labels):
    classList=[x[-1] for x in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat, bestValue = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [x[bestFeat] for x in dataSet]
    uniqueVals = set(featValues)
    if(is_number(dataSet[0][bestFeat])):
        subLabels = labels[:]
        rightDataSet, leftDateSet = splitDataSet_num(dataSet, bestFeat, bestValue)
        myTree[bestFeatLabel]['小于'+bestValue] = createTree(rightDataSet, subLabels)
        myTree[bestFeatLabel]['大于'+bestValue] = createTree(leftDateSet, subLabels)
    else:
        del (labels[bestFeat])
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

if __name__=='__main__':
    trainData=file2matrix('data.txt')
    labels=trainData[0]
    del (trainData[0])
    myTree=createTree(trainData,labels)
    treePlotter.createPlot(myTree)