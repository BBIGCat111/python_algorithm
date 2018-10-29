# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:25:14 2018

@author: 12100
"""
from numpy import *
import matplotlib.pyplot as plt

#导入数据进入数组
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #数据存入数组
        labelMat.append(int(lineArr[2]))                          
    return dataMat,labelMat

#计算的值为x代入θ为参数的逻辑函数1/(1+exp(-θT*x))的概率值
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn,classLabels):
    #转化为NunPy矩阵数据类型
    dataMatrix = mat(dataMatIn)     #变为numpy型的向量，m(样本数)行三列向量
    labelMat = mat(classLabels).transpose()  #转置前为一行m列向量，转置后为m行1列向量
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  #矩阵相乘      新的weights与所有样本继续计算新的h
        error = (labelMat - h)     #m行一列向量
        weights = weights + alpha * dataMatrix.transpose() * error   #转置保证前后向量列数和行数相等
    return weights


#画图函数
def plotBestFit(weights):  #weights.getA()必不可少
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
  
#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix) 
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))   #一个样本代入算h
        error = classLabels[i] - h                #一个样本y算error
        weights = weights + alpha * error * dataMatrix[i]    #得出一个新的weights，再依次代入新的样本计算
    return weights
       
#改进的随机梯度上升算法    
def stocGradAscent1(dataMatrix,classLabels,numIter=150):  #迭代数可以修改
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    #alpha每次迭代时需要调整
            randIndex = int(random.uniform(0,len(dataIndex)))   #随机选取更新,减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights 
    
#利用Logistic回归分类函数进行分类
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f")  % errorRate
    return errorRate

#本函数调用十次colicTest()函数并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f")  % (numTests,errorSum/float(numTests))
    
    
    
    
    
    
    
    
    
    
    
    
    
    










