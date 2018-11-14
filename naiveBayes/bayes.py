# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/11/12
@description: 使用朴素贝叶斯进行文本分类,easy,对一些简单的文本分类十分简单有用，如：垃圾邮件/辱骂型文本检测等等。
              就跟词频统计似的。。。
"""
import numpy as np

def loadDataSet():                                    # 创建实验样本
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr''licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    classVec = [0,1,0,1,0,1] # 1侮辱，0正常
    return postingList,classVec

def createVocabList(dataSet):                        # 小技巧：统计列表中出现过的词
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)          # set的并操作
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):              # 将输入中出现过的词进行标注
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('单词{}未出现于词典中'.format(word))
    return returnVec

def trainNB0(trainMatrix,trainCategory):               # trainMatrix 是一个稀疏矩阵，出现过的词为1，未出现为0
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 侮辱性文章概率
    p0Num = np.ones(numWords)                         # 出现词统计(防止概率为0)
    p1Num = np.ones(numWords)
    p0Denom = 2.0                                      # 词总数
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                   #防止乘积太小，造成下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return  1
    else:
        return  0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(classifyNB(thisDoc,p0V,p1V,pAb))

if __name__ == '__main__':
    testingNB()