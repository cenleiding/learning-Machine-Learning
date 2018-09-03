# coding:utf-8

"""
@author : CLD
@time:2018/8/2820:01
@description: K-nearest neighbors
            ~手写系统识别系统~
            但运行速度十分慢，每次识别：1*2000*1024
            且还要给出2M的空间存放训练集
"""

import numpy as np
import os
import KNN

def img2vector(filename):                            #将32*32的图像转换为1*1024的向量
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/'+fileNameStr)
    testFileList=os.listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/'+fileNameStr)
        classifierResult=KNN.classify0(vectorUnderTest,trainingMat,hwLabels,3)
        if classifierResult!=classNumStr: errorCount+=1
    print("错误率：%f" % (errorCount/mTest))

if __name__=='__main__':
    handwritingClassTest()