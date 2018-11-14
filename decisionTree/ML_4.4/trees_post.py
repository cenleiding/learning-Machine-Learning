# coding:utf-8

"""
@author : CLD
@time:2018/9/1118:49
@description:
              ML 习题4.4
              实现基尼指数，后减枝，不考虑连续特征
"""
from decisionTree import treePlotter


def file2matrix(filename):
    data=[]
    with open(filename,'r',encoding='utf-8') as fr:
        for line in fr.readlines():
           data.append([d for d in line.strip().split(',')])
    return data

def giniIndex(dataSet):
    num=len(dataSet)
    labelCount={}
    gini=1
    for line in dataSet:
        labelCount[line[-1]]=1 if line[-1] not in labelCount.keys() else labelCount[line[-1]]+1
    for label in labelCount.values():
        gini-=(label/num)**2
    return gini

def majorityCnt(dataSet):
    classCount={}
    for line in dataSet:
        classCount[line[-1]]=1 if line[-1] not in classCount.keys() else classCount[line[-1]]+1
    sortedClassCount=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] ==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGini = giniIndex(dataSet)
    bestFeature = -1
    for i in range(numFeatures):
        uniqueFeat=set([x[i] for x in dataSet])
        newGini=0.0
        for feat in uniqueFeat:
            subDataSet=splitDataSet(dataSet,i,feat)
            newGini+=len(subDataSet)/len(dataSet)*giniIndex(subDataSet)
        if newGini<bestGini:
            bestGini=newGini
            bestFeature=i
    return bestFeature

def postpruning(inputTree,trainData,testData,featLabels):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    if type(secondDict).__name__=='str':
        return False
    flag=True
    for key in secondDict.keys():                     #判断子节点是否都为叶节点
        if type(secondDict[key]).__name__=='dict':
            flag=False
    if flag:
        classCount = {}
        for line in trainData:
            classCount[line[-1]] = 1 if line[-1] not in classCount.keys() else classCount[line[-1]] + 1
        maxClass = sorted(classCount.items(), key=lambda x: x[1], reverse=True)[0][0]
        count = [x[-1] for x in testData].count(maxClass)
        oldAccuracy = count / len(testData)
        #
        count=0.0
        for line in testData:
            if secondDict[line[featIndex]]==line[-1]:
                count+=1
        newAccuracy = count/len(testData)
        if oldAccuracy>newAccuracy:
            return maxClass
        return 0
    else:
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                subLabels = featLabels[:]
                subLabels.remove(firstStr)
                info=postpruning(secondDict[key],splitDataSet(trainData,featIndex,key),splitDataSet(testData,featIndex,key),subLabels)
                if info:
                    secondDict[key]=info
    return 0

def createTree(trainDataSet, labels):
    classList=[c[-1] for c in trainDataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(trainDataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(trainDataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    for feature in set([x[bestFeat] for x in trainDataSet]):
        subLabels=labels[:]
        myTree[bestFeatLabel][feature]=createTree(splitDataSet(trainDataSet, bestFeat, feature),subLabels)
    return myTree

if __name__=='__main__':
    trainData=file2matrix('trainData.txt')
    testData=file2matrix('testData.txt')
    labels = trainData[0]
    del (trainData[0])
    del (testData[0])
    myTree=createTree(trainData.copy(),labels.copy())
    postpruning(myTree,trainData,testData,labels)
    treePlotter.createPlot(myTree)

