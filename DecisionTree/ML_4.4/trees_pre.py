# coding:utf-8

"""
@author : CLD
@time:2018/9/1118:49
@description:
              ML 习题4.4
              实现基尼指数，预减枝，不考虑连续特征
"""
from DecisionTree import treePlotter


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

def prepruning(trainDataSet,testDataSet, bestFeat):                       #预减枝
    #
    classCount={}
    for line in trainDataSet:
        classCount[line[-1]]=1 if line[-1] not in classCount.keys() else classCount[line[-1]]+1
    maxClass=sorted(classCount.items(),key=lambda x:x[1],reverse=True)[0][0]
    count=[x[-1] for x in testDataSet].count(maxClass)
    oldAccuracy=count/len(testDataSet)
    #
    uniqueFeat = set([x[bestFeat] for x in trainDataSet])
    count = 0.0
    for feat in uniqueFeat:
        classCount={}
        for line in trainDataSet:
            if line[bestFeat]==feat:
                classCount[line[-1]] = 1 if line[-1] not in classCount.keys() else classCount[line[-1]] + 1
        maxClass = sorted(classCount.items(), key=lambda x: x[1], reverse=True)[0][0]
        for line in testDataSet:
            if line[bestFeat]==feat and line[-1]==maxClass:
                count+=1
    newAccuracy=count/len(testDataSet)
    if newAccuracy>oldAccuracy:
        return True
    else:
        return False

def createTree(trainDataSet, testDataSet, labels):
    classList=[c[-1] for c in trainDataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(trainDataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(trainDataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    if prepruning(trainDataSet,testDataSet, bestFeat):
        for feature in set([x[bestFeat] for x in trainDataSet]):
            subLabels=labels[:]
            myTree[bestFeatLabel][feature]=createTree(splitDataSet(trainDataSet, bestFeat, feature), splitDataSet(testDataSet,bestFeat,feature),subLabels)
    else:
        classCount={}
        for line in trainDataSet:
            classCount[line[-1]] = 1 if line[-1] not in classCount.keys() else classCount[line[-1]] + 1
        maxClass = sorted(classCount.items(), key=lambda x: x[1], reverse=True)[0][0]
        return maxClass
    return myTree

if __name__=='__main__':
    trainData=file2matrix('trainData.txt')
    testData=file2matrix('testData.txt')
    labels = trainData[0]
    del (trainData[0])
    del (testData[0])
    myTree=createTree(trainData,testData,labels)
    treePlotter.createPlot(myTree)

