# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:48:33 2018

@author: 12100
c"""

from math import log
import operator

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #计算数据的个数，即行数
    labelCounts = {}            #创建字典 
    for featVec in dataSet:    #featVec表示每一个数据
        currentLabel = featVec[-1]     #currentLabel为键，给标签值, 为类别
        if currentLabel not in labelCounts.keys():  #起初字典无键
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1      #这一类的种类数量加一
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  
        shannonEnt -= prob * log(prob,2)  
    return shannonEnt
 

#创建数据集例子举例       
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']   #第一个特征对应dataSet[0],第二个特征对应dataSet[1]  
    return dataSet,labels

#按照给定特征划分数据集
 #根据axis等于value的特征将数据提出
def splitDataSet(dataSet,axis,value):    #axis为最好特征值的索引值，value为特征的特征值
    retDataSet = []         #创建新的list对象
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet   #返回的是根据某特征的特征值分出的数据集，且已经删去了这个特征值


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):    #看不懂
    numFeatures = len(dataSet[0]) - 1        #剩下的是特征的个数
    baseEntropy = calcShannonEnt(dataSet)    #计算数据集的熵，放到baseEntropy中
    bestInfoGain = 0.0                      #初始化熵增益
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]   #featList存储对应特征所有可能得取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:          #下面是计算每种划分方式的信息熵,特征i个，每个特征value个值
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))        #特征样本在总样本中的权重
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy        #计算i个特征的信息熵
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                 #返回的是一个特征值的索引号

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote] = 0
        classCount[vote] += 1
    #利用operator操作键值排序字典，并返回出现次数最多的分类名称
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建数的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]    #将最后一行的数据放到classList中，classList=['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0]) == len(classList):     #类别完全相同不需要再划分  ,数第一个种类的数目与总长度相同
        return classList[0]
    if len(dataSet[0]) == 1:                     #就是说特征数为1的时候
        return majorityCnt(classList)                 #就返回这个特征就行了，因为就这一个特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}       #创建字典
    del(labels[bestFeat])               #删除这个特征标签
    featValues = [example[bestFeat] for example in dataSet]   #把这个特征的所有值存到featValues
    uniqueVals = set(featValues)       #把这个特征作为集合，集合中为这个特征的值的种类的数量
    for value in uniqueVals:      #根据这个特征的所有种类的值求新的树
        subLabels = labels[:]     #减少特征的标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)  #bestFeat是标签索引号(即键），value是特征值,得出的是一个种类
    return myTree



def classify(inputTree, featLabels, testVec):    #测试函数，返回测试数据的类别
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      #将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



#使用Ppickle模块存储决策树
def storeTree(inputTree, filename):      #把数据文件的决策树储存在硬盘
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    
    
def grabTree(filename):            #此函数利用文件直接输出决策树，不需要下次再重新生成决策树
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)



















