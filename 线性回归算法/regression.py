# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:38:59 2018

@author: 12100
"""

from numpy import *
from time import sleep
import json
import requests

#第一个函数用来打开一个用tab键分隔的文本文件，这里仍然默认文件每行的最后一个值为目标值
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  #算出每行有三个数 - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):          #每行有3个数，但是只循环两次，把前两个数放进x矩阵，最后一个当y矩阵
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))  #把最后一个数当做y值
    return dataMat,labelMat

#第二个函数用来计算最佳拟合直线
def standRegres(xArr,yArr):  
    xMat = mat(xArr)     #读入x和y并将它们保存到矩阵中
    yMat = mat(yArr).T
    xTx = xMat.T*xMat    #计算xTx的值
    if linalg.det(xTx) == 0.0:   #检查行列式是否为0，看是否有逆矩阵， 必须要有逆矩阵
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I*(xMat.T*yMat)   #计算权重w = (xTx)^(-1)*xTy 的值
    return ws

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):   #其中testPoint为待测点，k值为0.01时较好，小于0.01过拟合
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]      # x0-x
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))   #w=exp(-|x0-x|/2*k**2)
    xTx = xMat.T * (weights * xMat)     
    if linalg.det(xTx) ==0.0:   #判断是否行列式等于0，有无逆矩阵
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))      #计算出回归系数
    return testPoint * ws        #此处的预测y值为testPoint * ws

#测试函数
def lwlrTest(testArr,xArr,yArr,k=1.0):   #testArr为测试数据
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)  #求解每个点的预测y值
    return yHat

#示例：  查看预测效果的误差  进行差的平方的和计算，得出方差来判断误差大小
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return 
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr,yArr):      #用于上一组的测试
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    #数据标准化
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat


#购物信息的获取
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'http://www.googleapis.com/shopping/search/v1/public/products? \
                key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr,setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlg,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except: 
            print("problem with item %d"  %  i)
                
def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,269.99)
    searchForSet(retX,retY,10179,2007,5195,499.99)
    searchForSet(retX,retY,10181,2007,3428,199.99)
    searchForSet(retX,retY,10189,2008,5922,299.99)
    searchForSet(retX,retY,10196,2009,3263,249.99)

 












